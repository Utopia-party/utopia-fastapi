import time
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from sqlalchemy import text

from core.config import settings
from core.database import AsyncSessionLocal, Base, engine
from models.admin import ActivityLog
from models.party import PartyNotice  # noqa: F401 — Base.metadata 등록용

from routers import admin, assets, auth, behavior_captcha, captcha, chat, notifications, parties, report, ws_notifications, payments, admin_handocr, praises, search, siteverify
from routers.settlement import router as settlement_router

from routers.mypage import profile, trust_history
from routers.user import referrers
from routers.mypage import parties as mypage_parties
from routers.mypage import payments as mypage_payments

from routers.quick_match import router as quick_match_router

from routers.admin_moderation_config import router as admin_mod_config_router
from routers.admin.cloud_monitor import router as admin_cloud_monitor_router
from routers.admin.saas_admin import router as admin_saas_router
from routers.admin.saas_admin_v2 import router as admin_saas_v2_router
from routers.admin import admin_quick_match
from routers.appeal import router as appeal_router
from routers.developer import router as developer_router
from routers.developer_v2 import router as developer_v2_router
from routers.saas_chat import router as saas_chat_router
from routers.saas_captcha_l2 import router as saas_captcha_l2_router

logging.basicConfig(level=logging.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'services'
                          AND column_name = 'original_price'
                    ) THEN
                        EXECUTE 'ALTER TABLE services ADD COLUMN original_price INTEGER';
                    END IF;

                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'services'
                          AND column_name = 'selling_price'
                    ) THEN
                        EXECUTE 'UPDATE services SET monthly_price = COALESCE(selling_price, monthly_price)';
                        EXECUTE 'ALTER TABLE services DROP COLUMN selling_price';
                    END IF;

                    EXECUTE 'UPDATE services SET original_price = COALESCE(original_price, monthly_price) WHERE original_price IS NULL';

                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'payments'
                          AND column_name = 'cancel_reason'
                    ) AND NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'payments'
                          AND column_name = 'status_reason'
                    ) THEN
                        EXECUTE 'ALTER TABLE payments RENAME COLUMN cancel_reason TO status_reason';
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'payments'
                          AND column_name = 'status_reason'
                    ) THEN
                        EXECUTE 'ALTER TABLE payments ADD COLUMN status_reason VARCHAR(200)';
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'admin_roles'
                          AND column_name = 'can_manage_payments'
                    ) THEN
                        EXECUTE 'ALTER TABLE admin_roles ADD COLUMN can_manage_payments BOOLEAN NOT NULL DEFAULT false';
                    END IF;

                    EXECUTE '
                        UPDATE admin_roles
                        SET can_manage_payments = true
                        WHERE can_manage_admins = true
                           OR can_approve_settlements = true
                    ';

                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'admin_roles'
                          AND column_name = 'can_approve_receipts'
                    ) AND NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'admin_roles'
                          AND column_name = 'can_manage_captcha'
                    ) THEN
                        EXECUTE '
                            ALTER TABLE admin_roles
                            RENAME COLUMN can_approve_receipts TO can_manage_captcha
                        ';
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'admin_roles'
                          AND column_name = 'can_manage_captcha'
                    ) THEN
                        EXECUTE '
                            ALTER TABLE admin_roles
                            ADD COLUMN can_manage_captcha BOOLEAN NOT NULL DEFAULT false
                        ';
                    END IF;

                    IF NOT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'admin_roles'
                          AND column_name = 'can_manage_handocr'
                    ) THEN
                        EXECUTE 'ALTER TABLE admin_roles ADD COLUMN can_manage_handocr BOOLEAN NOT NULL DEFAULT false';
                    END IF;

                    EXECUTE '
                        UPDATE admin_roles
                        SET can_manage_handocr = true
                        WHERE can_manage_admins = true
                           OR can_manage_captcha = true
                    ';

                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'activity_logs'
                          AND column_name = 'ip_address'
                          AND udt_name <> 'inet'
                    ) THEN
                        EXECUTE '
                            ALTER TABLE activity_logs
                            ALTER COLUMN ip_address TYPE inet
                            USING CASE
                                WHEN ip_address IS NULL OR btrim(ip_address::text) = '''' THEN NULL
                                ELSE ip_address::inet
                            END
                        ';
                    END IF;

                    -- payment_deadline 컬럼 추가 (파티 인원 충족 시 결제 마감일)
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'parties'
                          AND column_name = 'payment_deadline'
                    ) THEN
                        EXECUTE 'ALTER TABLE parties ADD COLUMN payment_deadline TIMESTAMPTZ';
                    END IF;

                    -- party_notices 테이블 생성 (채팅방 상단 공지)
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public'
                          AND table_name = 'party_notices'
                    ) THEN
                        EXECUTE '
                            CREATE TABLE party_notices (
                                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                party_id UUID UNIQUE NOT NULL REFERENCES parties(id) ON DELETE CASCADE,
                                created_by UUID NOT NULL REFERENCES users(id),
                                content TEXT NOT NULL,
                                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                            )
                        ';
                    END IF;

                END
                $$;
                """
            )
        )
    yield

from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

app = FastAPI(
    title="Party-Up API",
    description="파티업 백엔드 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

def _extract_actor_user_id(request: Request) -> uuid.UUID | None:
    access_token = request.cookies.get("access_token")
    if not access_token:
        return None

    try:
        payload = jwt.decode(access_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "access":
            return None
        return uuid.UUID(payload.get("sub", ""))
    except (JWTError, ValueError):
        return None

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    if request.headers.get("upgrade") == "websocket":
        return await call_next(request)
        
    try:
        return await call_next(request)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        headers = {}
        origin = request.headers.get("origin")
        if origin and origin in settings.ALLOWED_ORIGINS:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            headers["Vary"] = "Origin"
        return JSONResponse(status_code=500, content={"detail": str(e)}, headers=headers)

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    if request.headers.get("upgrade") == "websocket":
        return await call_next(request)

    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if elapsed_ms >= 200:
        print(f"[TIMING] {elapsed_ms:7.0f}ms  {request.method} {request.url.path}")
    return response



_ADMIN_PATH_LABELS: list[tuple[str, str, str]] = [
    ("GET",    "/api/admin/dashboard",                          "대시보드 조회"),
    ("GET",    "/api/admin/users",                              "사용자 목록 조회"),
    ("GET",    "/api/admin/users/",                             "사용자 상세 조회"),
    ("PATCH",  "/api/admin/users/",                             "사용자 정보 수정"),
    ("PATCH",  "/api/admin/users/{id}/status",                  "사용자 상태 변경"),
    ("PATCH",  "/api/admin/users/{id}/trust-score",             "신뢰도 점수 수정"),
    ("PATCH",  "/api/admin/users/{id}/recommender",             "추천인 수정"),
    ("GET",    "/api/admin/users/{id}/status-logs",             "사용자 상태 변경 이력 조회"),
    ("GET",    "/api/admin/users/{id}/access-logs",             "사용자 접근 로그 조회"),
    ("GET",    "/api/admin/services",                           "구독 서비스 목록 조회"),
    ("PATCH",  "/api/admin/services/",                          "구독 서비스 정보 수정"),
    ("GET",    "/api/admin/parties",                            "파티 목록 조회"),
    ("POST",   "/api/admin/parties/",                           "파티 강제 종료"),
    ("GET",    "/api/admin/parties/{id}/members",               "파티 멤버 목록 조회"),
    ("POST",   "/api/admin/parties/{id}/members/{uid}/kick",    "파티 멤버 강제 퇴장"),
    ("PATCH",  "/api/admin/parties/{id}/members/{uid}/role",    "파티 멤버 역할 변경"),
    ("GET",    "/api/admin/quick-match/requests",               "빠른매칭 요청 목록 조회"),
    ("GET",    "/api/admin/quick-match/requests/",              "빠른매칭 요청 상세 조회"),
    ("GET",    "/api/admin/quick-match/policy",                 "빠른매칭 정책 조회"),
    ("PATCH",  "/api/admin/quick-match/policy",                 "빠른매칭 정책 수정"),
    ("POST",   "/api/admin/quick-match/",                       "빠른매칭 관리 작업 수행"),
    ("GET",    "/api/admin/reports",                            "신고 목록 조회"),
    ("PATCH",  "/api/admin/reports/",                           "신고 처리"),
    ("GET",    "/api/admin/reports/evidences/",                 "신고 증거 파일 조회"),
    ("GET",    "/api/admin/moderation/chat-logs",               "채팅 모더레이션 로그 조회"),
    ("GET",    "/api/admin/moderation/chat-stats",              "채팅 모더레이션 통계 조회"),
    ("GET",    "/api/admin/moderation/chat-trend",              "채팅 모더레이션 추세 조회"),
    ("PATCH",  "/api/admin/moderation/chat-logs/",              "채팅 모더레이션 상태 변경"),
    ("GET",    "/api/admin/moderation/config",                  "모더레이션 설정 조회"),
    ("PATCH",  "/api/admin/moderation/config",                  "모더레이션 설정 수정"),
    ("POST",   "/api/admin/moderation/config/reset",            "모더레이션 설정 초기화"),
    ("POST",   "/api/admin/moderation/whitelist",               "모더레이션 허용 단어 추가"),
    ("DELETE", "/api/admin/moderation/whitelist/",              "모더레이션 허용 단어 삭제"),
    ("POST",   "/api/admin/moderation/blacklist",               "모더레이션 금지 단어 추가"),
    ("DELETE", "/api/admin/moderation/blacklist/",              "모더레이션 금지 단어 삭제"),
    ("GET",    "/api/admin/moderation/finetune/stats",          "모더레이션 파인튜닝 통계 조회"),
    ("POST",   "/api/admin/moderation/unblock/user/",           "채팅 차단 사용자 해제"),
    ("GET",    "/api/admin/moderation/chat-bans",               "채팅 차단 목록 조회"),
    ("GET",    "/api/admin/users/{id}/status-logs",             "사용자 상태 로그 조회"),
    ("GET",    "/api/admin/captcha/shadow",                     "캡챠 섀도우 설정 조회"),
    ("PUT",    "/api/admin/captcha/shadow",                     "캡챠 섀도우 설정 변경"),
    ("GET",    "/api/admin/captcha/blocked-ips",                "캡챠 차단 IP 목록 조회"),
    ("DELETE", "/api/admin/captcha/blocked-ips/",               "캡챠 차단 IP 삭제"),
    ("DELETE", "/api/admin/captcha/blocked-ips",                "캡챠 차단 IP 전체 삭제"),
    ("GET",    "/api/admin/captcha/config",                     "캡챠 설정 조회"),
    ("PUT",    "/api/admin/captcha/config",                     "캡챠 설정 변경"),
    ("POST",   "/api/admin/captcha/force-challenge",            "캡챠 강제 발동"),
    ("GET",    "/api/admin/captcha/stats",                      "캡챠 통계 조회"),
    ("GET",    "/api/admin/captcha/sessions",                   "캡챠 세션 목록 조회"),
    ("GET",    "/api/admin/captcha/sessions/",                  "캡챠 세션 이미지 조회"),
    ("GET",    "/api/admin/captcha/images",                     "캡챠 이미지 목록 조회"),
    ("GET",    "/api/admin/captcha/images/",                    "캡챠 이미지 세트 조회"),
    ("PUT",    "/api/admin/captcha/sets/",                      "캡챠 세트 비활성화"),
    ("PUT",    "/api/admin/captcha/images/batch-deactivate",    "캡챠 이미지 일괄 비활성화"),
    ("PUT",    "/api/admin/captcha/images/",                    "캡챠 이미지 비활성화"),
    ("POST",   "/api/admin/captcha/generate",                   "캡챠 이미지 생성"),
    ("GET",    "/api/admin/captcha/generate/status",            "캡챠 이미지 생성 상태 조회"),
    ("GET",    "/api/admin/handocr/records",                    "HandOCR 인증 기록 조회"),
    ("GET",    "/api/admin/handocr/health",                     "HandOCR 서비스 상태 조회"),
    ("GET",    "/api/admin/handocr/image",                      "HandOCR 이미지 조회"),
    ("GET",    "/api/admin/handocr/blocks",                     "HandOCR 차단 목록 조회"),
    ("POST",   "/api/admin/handocr/blocks/",                    "HandOCR 차단 IP 해제"),
    ("POST",   "/api/admin/handocr/ips/",                       "HandOCR IP 실패 횟수 초기화"),
    ("GET",    "/api/admin/handocr/sessions",                   "HandOCR 세션 목록 조회"),
    ("POST",   "/api/admin/handocr/sessions/",                  "HandOCR 세션 만료 처리"),
    ("GET",    "/api/admin/settlements",                        "정산 목록 조회"),
    ("PATCH",  "/api/admin/settlements/",                       "정산 승인/거절 처리"),
    ("GET",    "/api/admin/payments",                           "수익 내역 조회"),
    ("GET",    "/api/admin/receipts",                           "영수증 목록 조회"),
    ("PATCH",  "/api/admin/receipts/",                          "영수증 상태 변경"),
    ("GET",    "/api/admin/appeals",                            "이의제기 목록 조회"),
    ("PATCH",  "/api/admin/appeals/",                           "이의제기 처리"),
    ("GET",    "/api/admin/me",                                 "내 관리자 권한 조회"),
    ("GET",    "/api/admin/roles",                              "관리자 권한 목록 조회"),
    ("PUT",    "/api/admin/roles/",                             "관리자 권한 수정"),
    ("DELETE", "/api/admin/roles/",                             "관리자 권한 삭제"),
    ("GET",    "/api/admin/cloud-monitor/summary",              "클라우드 모니터링 요약 조회"),
    ("GET",    "/api/admin/cloud-monitor/range",                "클라우드 모니터링 범위 조회"),
    ("GET",    "/api/admin/cloud-monitor/lb",                   "로드밸런서 상태 조회"),
    ("GET",    "/api/admin/cloud-monitor/debug/labels",         "클라우드 모니터링 디버그 레이블 조회"),
    ("GET",    "/api/admin/cloud-monitor/debug/raw",            "클라우드 모니터링 디버그 원본 조회"),
    ("GET",    "/api/admin/cloud-monitor/debug/disk-unit",      "클라우드 모니터링 디스크 단위 조회"),
    # SaaS v2
    ("GET",    "/api/admin/saas-v2/keys",                       "SaaS v2 키 목록 조회"),
    ("POST",   "/api/admin/saas-v2/keys",                       "SaaS v2 키 발급"),
    ("GET",    "/api/admin/saas-v2/keys/",                      "SaaS v2 키 상세 조회"),
    ("PUT",    "/api/admin/saas-v2/keys/",                      "SaaS v2 키 수정"),
    ("POST",   "/api/admin/saas-v2/keys/",                      "SaaS v2 키 액션"),
    ("GET",    "/api/admin/saas-v2/stats",                      "SaaS v2 통계 조회"),
]


def _admin_path_label(method: str, path: str) -> str:
    for m, p, label in _ADMIN_PATH_LABELS:
        if m == method and not p.endswith("/") and p == path:
            return label
    for m, p, label in _ADMIN_PATH_LABELS:
        if m == method and p.endswith("/") and path.startswith(p):
            return label
    return f"관리자 API 접근 ({method} {path})"


@app.middleware("http")
async def admin_access_log_middleware(request: Request, call_next):
    if request.headers.get("upgrade") == "websocket":
        return await call_next(request)

    response = await call_next(request)

    if request.url.path.startswith("/api/admin") and request.url.path != "/api/admin/logs":
        try:
            async with AsyncSessionLocal() as session:
                session.add(
                    ActivityLog(
                        actor_user_id=_extract_actor_user_id(request),
                        action_type="관리자 접근",
                        description=_admin_path_label(request.method, request.url.path),
                        ip_address=request.client.host if request.client else None,
                        extra_metadata={"path": request.url.path, "method": request.method, "status": response.status_code},
                    )
                )
                await session.commit()
        except Exception:
            logging.exception("관리자 접근 로그 기록 실패")

    return response

app.include_router(auth.router, prefix="/api")
app.include_router(parties.router, prefix="/api")
app.include_router(payments.router, prefix="/api")
app.include_router(settlement_router, prefix="/api")
app.include_router(quick_match_router, prefix="/api")
app.include_router(notifications.router, prefix="/api")
app.include_router(ws_notifications.router)
app.include_router(chat.router, prefix="/api")
app.include_router(behavior_captcha.router, prefix="/api")  
app.include_router(captcha.router, prefix="/api")
app.include_router(siteverify.router) 
app.include_router(assets.router, prefix="/api", tags=["Assets"])

app.include_router(admin.router, prefix="/api")  
app.include_router(report.router, prefix="/api")  

app.include_router(profile.router, prefix="/api")
app.include_router(mypage_parties.router, prefix="/api")
app.include_router(trust_history.router, prefix="/api")
app.include_router(mypage_payments.router, prefix="/api")
app.include_router(referrers.router, prefix="/api")

app.include_router(admin_mod_config_router, prefix="/api")
app.include_router(admin_cloud_monitor_router, prefix="/api")
app.include_router(admin_saas_router, prefix="/api")
app.include_router(admin_saas_v2_router, prefix="/api")
app.include_router(admin_quick_match.router, prefix="/api")
app.include_router(appeal_router)
app.include_router(developer_router, prefix="/api")
app.include_router(developer_v2_router, prefix="/api")
app.include_router(saas_chat_router, prefix="/api")
app.include_router(saas_captcha_l2_router, prefix="/api")

app.include_router(admin_handocr.router, prefix="/api")

app.include_router(praises.router, prefix='/api')

app.include_router(search.router, prefix='/api')

_sdk_dir = Path(__file__).resolve().parent / "sdk"
if _sdk_dir.is_dir():
    app.mount("/sdk", StaticFiles(directory=str(_sdk_dir)), name="sdk")

@app.get("/api/health")
async def health():
    return {"status": "ok"}
