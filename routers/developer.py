"""
SaaS API 키 관리 — 일반 사용자의 자신의 API 키 조회/발급/수정

사용자가 로그인 후:
  - 자신의 API 키 목록 조회
  - 새 API 키 발급 (Free 플랜, 월 1,000 요청)
  - 자신의 키만 수정/재발급 가능
  - 사용 로그 및 통계 조회
"""
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import require_user
from models.user import User

router = APIRouter(prefix="/developer", tags=["developer"])


# ══════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════

class DeveloperKeyCreateRequest(BaseModel):
    client_name: str = Field(..., min_length=1, max_length=200)
    allowed_domains: Optional[list[str]] = None


class DeveloperKeyUpdateRequest(BaseModel):
    client_name: Optional[str] = None
    allowed_domains: Optional[list[str]] = None


class DeveloperKeyOut(BaseModel):
    id: str
    client_name: str
    api_key: str
    secret_key: str  # Masked for list/detail, full for create/rotate
    allowed_domains: Optional[list[str]]
    monthly_limit: int
    current_month_usage: int
    plan: str
    is_active: bool
    created_at: Optional[str]


class DeveloperKeyListResponse(BaseModel):
    total: int
    items: list[DeveloperKeyOut]


class UsageLogOut(BaseModel):
    id: str
    endpoint: str
    status_code: int
    response_time_ms: Optional[int]
    created_at: Optional[str]


class UsageLogListResponse(BaseModel):
    total: int
    items: list[UsageLogOut]


class UsageSummaryOut(BaseModel):
    total_keys: int
    active_keys: int
    total_usage_this_month: int


class PlanInquiryCreateRequest(BaseModel):
    desired_plan: str = Field(..., description="희망 플랜: starter, pro, enterprise")
    message: Optional[str] = Field(None, max_length=500)


class PlanInquiryOut(BaseModel):
    id: str
    user_id: str
    user_email: Optional[str]
    desired_plan: str
    message: Optional[str]
    status: str  # pending, completed
    created_at: Optional[str]


class PlanInquiryListResponse(BaseModel):
    total: int
    items: list[PlanInquiryOut]


# ══════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════

def _generate_site_key() -> str:
    """pk_live_partyup_<16자리hex>"""
    return f"pk_live_partyup_{secrets.token_hex(16)}"


def _generate_secret_key() -> str:
    """sk_live_partyup_<32자리hex>"""
    return f"sk_live_partyup_{secrets.token_hex(32)}"


def _mask_secret_key(secret_key: str) -> str:
    """Secret key를 마스킹: 처음 15자 + ••••••••"""
    if len(secret_key) <= 15:
        return secret_key
    return secret_key[:15] + "••••••••"


def _fmt_dt(value) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


# ══════════════════════════════════════════════════════════════
# 1. 내 API 키 목록 조회
# ══════════════════════════════════════════════════════════════

@router.get("/keys", response_model=DeveloperKeyListResponse)
async def list_my_api_keys(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    """현재 사용자의 API 키 목록"""
    # 총 개수
    count_result = await db.execute(
        text("SELECT COUNT(*) FROM api_keys WHERE created_by = :user_id"),
        {"user_id": str(current_user.id)},
    )
    total = count_result.scalar() or 0

    # 데이터
    result = await db.execute(
        text("""
            SELECT id, client_name, api_key, secret_key, allowed_domains,
                   monthly_limit, current_month_usage, plan, is_active,
                   created_at
            FROM api_keys
            WHERE created_by = :user_id
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {
            "user_id": str(current_user.id),
            "limit": size,
            "offset": (page - 1) * size,
        },
    )

    items = []
    for row in result.mappings():
        items.append(DeveloperKeyOut(
            id=str(row["id"]),
            client_name=row["client_name"],
            api_key=row["api_key"],
            secret_key=_mask_secret_key(row["secret_key"]),
            allowed_domains=row["allowed_domains"],
            monthly_limit=row["monthly_limit"],
            current_month_usage=row["current_month_usage"],
            plan=row["plan"],
            is_active=row["is_active"],
            created_at=_fmt_dt(row.get("created_at")),
        ))

    return DeveloperKeyListResponse(total=total, items=items)


# ══════════════════════════════════════════════════════════════
# 2. API 키 발급 (생성)
# ══════════════════════════════════════════════════════════════

@router.post("/keys", response_model=DeveloperKeyOut, status_code=201)
async def create_my_api_key(
    payload: DeveloperKeyCreateRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """새 API 키 발급 (Free 플랜, 월 10,000 요청, 최대 3개)"""
    # 최대 3개 제한
    count_result = await db.execute(
        text("SELECT COUNT(*) FROM api_keys WHERE created_by = :user_id"),
        {"user_id": str(current_user.id)},
    )
    current_count = count_result.scalar() or 0
    if current_count >= 3:
        raise HTTPException(
            status_code=400,
            detail="API 키는 최대 3개까지 발급 가능합니다.",
        )

    site_key = _generate_site_key()
    secret_key = _generate_secret_key()

    result = await db.execute(
        text("""
            INSERT INTO api_keys (client_name, api_key, secret_key, allowed_domains,
                                  monthly_limit, plan, is_active, current_month_usage,
                                  created_by)
            VALUES (:client_name, :api_key, :secret_key, :allowed_domains,
                    :monthly_limit, 'free', true, 0, :created_by)
            RETURNING id, client_name, api_key, secret_key, allowed_domains,
                      monthly_limit, current_month_usage, plan, is_active,
                      created_at
        """),
        {
            "client_name": payload.client_name,
            "api_key": site_key,
            "secret_key": secret_key,
            "allowed_domains": payload.allowed_domains,
            "monthly_limit": 1000,
            "created_by": str(current_user.id),
        },
    )
    await db.commit()
    row = result.mappings().first()

    # 생성/재발급 시에는 secret_key 전체 노출
    return DeveloperKeyOut(
        id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=row["secret_key"],  # 마스킹 없음
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt_dt(row.get("created_at")),
    )


# ══════════════════════════════════════════════════════════════
# 3. 내 API 키 단건 조회
# ══════════════════════════════════════════════════════════════

@router.get("/keys/{key_id}", response_model=DeveloperKeyOut)
async def get_my_api_key(
    key_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 API 키 상세 조회 (본인 것만)"""
    result = await db.execute(
        text("""
            SELECT id, client_name, api_key, secret_key, allowed_domains,
                   monthly_limit, current_month_usage, plan, is_active,
                   created_at, created_by
            FROM api_keys WHERE id = :id
        """),
        {"id": key_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    # 본인 확인
    if str(row["created_by"]) != str(current_user.id):
        raise HTTPException(
            status_code=403,
            detail="다른 사용자의 API 키에 접근할 수 없습니다.",
        )

    return DeveloperKeyOut(
        id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=_mask_secret_key(row["secret_key"]),
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt_dt(row.get("created_at")),
    )


# ══════════════════════════════════════════════════════════════
# 4. 내 API 키 수정 (이름, 도메인만)
# ══════════════════════════════════════════════════════════════

@router.put("/keys/{key_id}", response_model=DeveloperKeyOut)
async def update_my_api_key(
    key_id: str,
    payload: DeveloperKeyUpdateRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 API 키 정보 수정 (이름, 도메인만)"""
    # 먼저 키가 본인 것인지 확인
    check_result = await db.execute(
        text("SELECT created_by FROM api_keys WHERE id = :id"),
        {"id": key_id},
    )
    check_row = check_result.mappings().first()
    if not check_row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    if str(check_row["created_by"]) != str(current_user.id):
        raise HTTPException(
            status_code=403,
            detail="다른 사용자의 API 키를 수정할 수 없습니다.",
        )

    set_parts = []
    params: dict = {"id": key_id}

    if payload.client_name is not None:
        set_parts.append("client_name = :client_name")
        params["client_name"] = payload.client_name
    if payload.allowed_domains is not None:
        set_parts.append("allowed_domains = :allowed_domains")
        params["allowed_domains"] = payload.allowed_domains

    if not set_parts:
        raise HTTPException(status_code=400, detail="변경할 항목이 없습니다.")

    set_parts.append("updated_at = NOW()")
    set_clause = ", ".join(set_parts)

    result = await db.execute(
        text(f"""
            UPDATE api_keys SET {set_clause}
            WHERE id = :id
            RETURNING id, client_name, api_key, secret_key, allowed_domains,
                      monthly_limit, current_month_usage, plan, is_active,
                      created_at
        """),
        params,
    )
    await db.commit()
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    return DeveloperKeyOut(
        id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=_mask_secret_key(row["secret_key"]),
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt_dt(row.get("created_at")),
    )


# ══════════════════════════════════════════════════════════════
# 5. Secret Key 재발급
# ══════════════════════════════════════════════════════════════

@router.post("/keys/{key_id}/rotate-secret", response_model=DeveloperKeyOut)
async def rotate_my_secret_key(
    key_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """Secret Key 재발급 (site_key는 유지, 본인 것만)"""
    # 먼저 키가 본인 것인지 확인
    check_result = await db.execute(
        text("SELECT created_by FROM api_keys WHERE id = :id"),
        {"id": key_id},
    )
    check_row = check_result.mappings().first()
    if not check_row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    if str(check_row["created_by"]) != str(current_user.id):
        raise HTTPException(
            status_code=403,
            detail="다른 사용자의 API 키의 Secret을 재발급할 수 없습니다.",
        )

    new_secret = _generate_secret_key()
    result = await db.execute(
        text("""
            UPDATE api_keys
            SET secret_key = :secret_key, updated_at = NOW()
            WHERE id = :id
            RETURNING id, client_name, api_key, secret_key, allowed_domains,
                      monthly_limit, current_month_usage, plan, is_active,
                      created_at
        """),
        {"id": key_id, "secret_key": new_secret},
    )
    await db.commit()
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    # 재발급 시에는 secret_key 전체 노출
    return DeveloperKeyOut(
        id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=row["secret_key"],  # 마스킹 없음
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt_dt(row.get("created_at")),
    )


# ══════════════════════════════════════════════════════════════
# 6. 사용 로그 조회
# ══════════════════════════════════════════════════════════════

@router.get("/keys/{key_id}/usage", response_model=UsageLogListResponse)
async def get_my_usage_logs(
    key_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
):
    """현재 사용자의 특정 API 키 사용 로그"""
    # 먼저 키가 본인 것인지 확인
    check_result = await db.execute(
        text("SELECT created_by FROM api_keys WHERE id = :id"),
        {"id": key_id},
    )
    check_row = check_result.mappings().first()
    if not check_row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    if str(check_row["created_by"]) != str(current_user.id):
        raise HTTPException(
            status_code=403,
            detail="다른 사용자의 API 키 로그에 접근할 수 없습니다.",
        )

    count_result = await db.execute(
        text("SELECT COUNT(*) FROM api_usage_logs WHERE api_key_id = :key_id"),
        {"key_id": key_id},
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        text("""
            SELECT id, endpoint, status_code, response_time_ms, created_at
            FROM api_usage_logs
            WHERE api_key_id = :key_id
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {"key_id": key_id, "limit": size, "offset": (page - 1) * size},
    )

    items = []
    for row in result.mappings():
        items.append(UsageLogOut(
            id=str(row["id"]),
            endpoint=row["endpoint"],
            status_code=row["status_code"],
            response_time_ms=row.get("response_time_ms"),
            created_at=_fmt_dt(row.get("created_at")),
        ))

    return UsageLogListResponse(total=total, items=items)


# ══════════════════════════════════════════════════════════════
# 7. 전체 사용량 요약
# ══════════════════════════════════════════════════════════════

@router.get("/usage-summary", response_model=UsageSummaryOut)
async def get_my_usage_summary(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """현재 사용자의 전체 API 사용량 요약"""
    result = await db.execute(
        text("""
            SELECT
                COUNT(*) as total_keys,
                COUNT(*) FILTER (WHERE is_active) as active_keys,
                COALESCE(SUM(current_month_usage), 0) as total_usage
            FROM api_keys
            WHERE created_by = :user_id
        """),
        {"user_id": str(current_user.id)},
    )
    row = result.mappings().first()

    return UsageSummaryOut(
        total_keys=row["total_keys"] or 0,
        active_keys=row["active_keys"] or 0,
        total_usage_this_month=row["total_usage"] or 0,
    )


# ══════════════════════════════════════════════════════════════
# 8. API 키 삭제 — 관리자 전용으로 이관 (saas_admin.py)
#    사용자가 삭제→재발급으로 사용량을 초기화하는 것을 방지
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# 9. 플랜 업그레이드 문의 생성
# ══════════════════════════════════════════════════════════════

@router.post(
    "/plan-inquiry",
    response_model=PlanInquiryOut,
    status_code=status.HTTP_201_CREATED,
    summary="플랜 업그레이드 문의",
)
async def create_plan_inquiry(
    payload: PlanInquiryCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """사용자가 플랜 업그레이드를 문의한다."""
    valid_plans = ("starter", "pro", "enterprise")
    if payload.desired_plan not in valid_plans:
        raise HTTPException(
            status_code=400,
            detail=f"유효한 플랜: {', '.join(valid_plans)}",
        )

    # 중복 대기 문의 방지
    dup = await db.execute(
        text("""
            SELECT id FROM plan_inquiries
            WHERE user_id = :uid AND status = 'pending'
        """),
        {"uid": str(current_user.id)},
    )
    if dup.mappings().first():
        raise HTTPException(
            status_code=400,
            detail="이미 대기 중인 문의가 있습니다.",
        )

    result = await db.execute(
        text("""
            INSERT INTO plan_inquiries (user_id, desired_plan, message, status)
            VALUES (:uid, :plan, :msg, 'pending')
            RETURNING id, user_id, desired_plan, message, status, created_at
        """),
        {
            "uid": str(current_user.id),
            "plan": payload.desired_plan,
            "msg": payload.message,
        },
    )
    await db.commit()
    row = result.mappings().first()

    return PlanInquiryOut(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        user_email=current_user.email,
        desired_plan=row["desired_plan"],
        message=row["message"],
        status=row["status"],
        created_at=_fmt_dt(row["created_at"]),
    )


# ══════════════════════════════════════════════════════════════
# 10. 내 플랜 문의 목록 조회
# ══════════════════════════════════════════════════════════════

@router.get(
    "/plan-inquiries",
    response_model=PlanInquiryListResponse,
    summary="내 플랜 문의 목록",
)
async def list_my_plan_inquiries(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_user),
):
    """사용자가 자신의 플랜 문의 내역을 확인한다."""
    result = await db.execute(
        text("""
            SELECT id, user_id, desired_plan, message, status, created_at
            FROM plan_inquiries
            WHERE user_id = :uid
            ORDER BY created_at DESC
        """),
        {"uid": str(current_user.id)},
    )
    rows = result.mappings().all()

    return PlanInquiryListResponse(
        total=len(rows),
        items=[
            PlanInquiryOut(
                id=str(r["id"]),
                user_id=str(r["user_id"]),
                user_email=current_user.email,
                desired_plan=r["desired_plan"],
                message=r["message"],
                status=r["status"],
                created_at=_fmt_dt(r["created_at"]),
            )
            for r in rows
        ],
    )
