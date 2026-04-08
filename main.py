import time
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt

from core.config import settings
from core.database import AsyncSessionLocal, Base, engine
from models.admin import ActivityLog
from routers import admin, auth, captcha, chat, notifications, parties, user_interests

logging.basicConfig(level=logging.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(
    title="Party-Up API",
    description="파티업 백엔드 API",
    version="1.0.0",
    lifespan=lifespan,
)


# 상원: 관리자 접근 로그를 남기기 전에 쿠키 access token에서 사용자 식별자를 안전하게 읽습니다.
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

# [중요] CORS 미들웨어를 먼저 등록하여 모든 요청에 대해 보안 헤더를 처리합니다.
app.add_middleware(
    CORSMiddleware,
    # 상원: 쿠키 기반 인증과 캡챠 API가 같이 동작하려면 와일드카드가 아니라 명시적 오리진 목록을 써야 합니다.
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 예외 처리 미들웨어 (WebSocket 제외 로직 포함)
@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    # 웹소켓 요청은 HTTP 미들웨어 로직을 타지 않도록 즉시 통과
    if request.headers.get("upgrade") == "websocket":
        return await call_next(request)
        
    try:
        return await call_next(request)
    except HTTPException:
        # 상원: FastAPI가 기본 처리하는 HTTP 예외는 그대로 넘겨야 CORS 헤더와 상태 코드가 유지됩니다.
        raise
    except Exception as e:
        traceback.print_exc()
        headers = {}
        origin = request.headers.get("origin")
        if origin and origin in settings.ALLOWED_ORIGINS:
            # 상원: 예외 응답도 프론트가 읽을 수 있도록 허용된 오리진이면 CORS 헤더를 직접 붙입니다.
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            headers["Vary"] = "Origin"
        return JSONResponse(status_code=500, content={"detail": str(e)}, headers=headers)

# 2. 응답 시간 측정 미들웨어 (WebSocket 제외 로직 포함)
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


# 상원: /api/admin 하위 요청은 응답이 끝난 뒤 접근 로그를 activity_logs 테이블에 남깁니다.
@app.middleware("http")
async def admin_access_log_middleware(request: Request, call_next):
    if request.headers.get("upgrade") == "websocket":
        return await call_next(request)

    response = await call_next(request)

    if request.url.path.startswith("/api/admin"):
        try:
            async with AsyncSessionLocal() as session:
                session.add(
                    ActivityLog(
                        actor_user_id=_extract_actor_user_id(request),
                        action_type="admin_access",
                        description=f"{request.method} {request.url.path} -> {response.status_code}",
                        path=request.url.path,
                        ip_address=request.client.host if request.client else None,
                    )
                )
                await session.commit()
        except Exception:
            logging.exception("관리자 접근 로그 기록 실패")

    return response

# 라우터 등록 (prefix="/api" 유지)
app.include_router(auth.router, prefix="/api")
app.include_router(parties.router, prefix="/api")
app.include_router(notifications.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(captcha.router, prefix="/api", tags=["Captcha"])
app.include_router(user_interests.router, prefix="/api")
# 상원: 관리자 페이지가 실제 데이터를 읽고 상태를 바꿀 수 있도록 관리자 라우터를 연결합니다.
app.include_router(admin.router, prefix="/api")  # 상원

@app.get("/api/health")
async def health():
    return {"status": "ok"}
