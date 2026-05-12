import secrets
from datetime import datetime
from typing import Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import require_user
from models.user import User

router = APIRouter(prefix="/developer-v2", tags=["developer-v2"])

ServiceType = Literal["captcha_l2", "chat_filter"]
MAX_KEYS_PER_SERVICE = 3


# ── Schemas ──────────────────────────────────────────────────────────────────

class KeyCreateRequest(BaseModel):
    service_type: ServiceType
    client_name: str = Field(..., min_length=1, max_length=200)
    allowed_domains: Optional[list[str]] = None


class KeyUpdateRequest(BaseModel):
    client_name: Optional[str] = None
    allowed_domains: Optional[list[str]] = None


class KeyOut(BaseModel):
    id: str
    service_type: str
    client_name: str
    api_key: str
    secret_key: str
    allowed_domains: Optional[list[str]]
    monthly_limit: int
    current_month_usage: int
    plan: str
    is_active: bool
    created_at: Optional[str]


class KeyListResponse(BaseModel):
    total: int
    items: list[KeyOut]


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
    service_type: ServiceType
    desired_plan: str = Field(..., pattern="^(starter|pro|enterprise)$")
    message: Optional[str] = Field(None, max_length=500)


class PlanInquiryOut(BaseModel):
    id: str
    user_id: str
    user_email: Optional[str]
    service_type: str
    desired_plan: str
    message: Optional[str]
    status: str
    created_at: Optional[str]


class PlanInquiryListResponse(BaseModel):
    total: int
    items: list[PlanInquiryOut]


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _gen_key(service_type: str) -> tuple[str, str]:
    prefix = "l2" if service_type == "captcha_l2" else "chat"
    return (
        f"pk_{prefix}_partyup_{secrets.token_hex(16)}",
        f"sk_{prefix}_partyup_{secrets.token_hex(32)}",
    )


def _mask(secret: str) -> str:
    return secret[:15] + "••••••••" if len(secret) > 15 else secret


def _fmt(value) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _to_out(row: dict, mask_secret: bool = True) -> KeyOut:
    return KeyOut(
        id=str(row["id"]),
        service_type=row["service_type"],
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=_mask(row["secret_key"]) if mask_secret else row["secret_key"],
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt(row.get("created_at")),
    )


async def _assert_owner(db: AsyncSession, key_id: str, user_id: str) -> None:
    result = await db.execute(
        text("SELECT created_by FROM saas_api_keys WHERE id = :id"),
        {"id": key_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="키를 찾을 수 없습니다.")
    if str(row["created_by"]) != user_id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다.")


# ── 1. 목록 조회 ──────────────────────────────────────────────────────────────

@router.get("/keys", response_model=KeyListResponse)
async def list_my_keys(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
    service_type: Optional[ServiceType] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    conditions = ["created_by = :user_id"]
    params: dict = {"user_id": str(current_user.id)}

    if service_type:
        conditions.append("service_type = :service_type")
        params["service_type"] = service_type

    where = " WHERE " + " AND ".join(conditions)

    total = (await db.execute(
        text(f"SELECT COUNT(*) FROM saas_api_keys{where}"), params
    )).scalar() or 0

    params["limit"] = size
    params["offset"] = (page - 1) * size

    result = await db.execute(
        text(f"""
            SELECT id, service_type, client_name, api_key, secret_key,
                   allowed_domains, monthly_limit, current_month_usage,
                   plan, is_active, created_at
            FROM saas_api_keys{where}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )

    return KeyListResponse(
        total=total,
        items=[_to_out(dict(r), mask_secret=True) for r in result.mappings()],
    )


# ── 2. 발급 ───────────────────────────────────────────────────────────────────

@router.post("/keys", response_model=KeyOut, status_code=201)
async def create_my_key(
    payload: KeyCreateRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    count = (await db.execute(
        text("""
            SELECT COUNT(*) FROM saas_api_keys
            WHERE created_by = :user_id AND service_type = :service_type
        """),
        {"user_id": str(current_user.id), "service_type": payload.service_type},
    )).scalar() or 0

    if count >= MAX_KEYS_PER_SERVICE:
        raise HTTPException(
            status_code=400,
            detail=f"{payload.service_type} 키는 최대 {MAX_KEYS_PER_SERVICE}개까지 발급 가능합니다.",
        )

    api_key, secret_key = _gen_key(payload.service_type)

    free_limit = 500 if payload.service_type == "captcha_l2" else 5000

    result = await db.execute(
        text("""
            INSERT INTO saas_api_keys
                (service_type, client_name, api_key, secret_key,
                 allowed_domains, monthly_limit, plan, is_active,
                 current_month_usage, created_by)
            VALUES
                (:service_type, :client_name, :api_key, :secret_key,
                 :allowed_domains, :monthly_limit, 'free', true, 0, :created_by)
            RETURNING id, service_type, client_name, api_key, secret_key,
                      allowed_domains, monthly_limit, current_month_usage,
                      plan, is_active, created_at
        """),
        {
            "service_type": payload.service_type,
            "client_name": payload.client_name,
            "api_key": api_key,
            "secret_key": secret_key,
            "allowed_domains": payload.allowed_domains,
            "monthly_limit": free_limit,
            "created_by": str(current_user.id),
        },
    )
    await db.commit()
    return _to_out(dict(result.mappings().first()), mask_secret=False)


# ── 3. 단건 조회 ──────────────────────────────────────────────────────────────

@router.get("/keys/{key_id}", response_model=KeyOut)
async def get_my_key(
    key_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_owner(db, key_id, str(current_user.id))

    result = await db.execute(
        text("""
            SELECT id, service_type, client_name, api_key, secret_key,
                   allowed_domains, monthly_limit, current_month_usage,
                   plan, is_active, created_at
            FROM saas_api_keys WHERE id = :id
        """),
        {"id": key_id},
    )
    return _to_out(dict(result.mappings().first()), mask_secret=True)


# ── 4. 수정 ───────────────────────────────────────────────────────────────────

@router.put("/keys/{key_id}", response_model=KeyOut)
async def update_my_key(
    key_id: str,
    payload: KeyUpdateRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_owner(db, key_id, str(current_user.id))

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
    result = await db.execute(
        text(f"""
            UPDATE saas_api_keys SET {', '.join(set_parts)}
            WHERE id = :id
            RETURNING id, service_type, client_name, api_key, secret_key,
                      allowed_domains, monthly_limit, current_month_usage,
                      plan, is_active, created_at
        """),
        params,
    )
    await db.commit()
    return _to_out(dict(result.mappings().first()), mask_secret=True)


# ── 5. Secret 재발급 ──────────────────────────────────────────────────────────

@router.post("/keys/{key_id}/rotate-secret", response_model=KeyOut)
async def rotate_my_secret(
    key_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_owner(db, key_id, str(current_user.id))

    svc = (await db.execute(
        text("SELECT service_type FROM saas_api_keys WHERE id = :id"),
        {"id": key_id},
    )).mappings().first()

    _, new_secret = _gen_key(svc["service_type"])
    result = await db.execute(
        text("""
            UPDATE saas_api_keys
            SET secret_key = :secret_key, updated_at = NOW()
            WHERE id = :id
            RETURNING id, service_type, client_name, api_key, secret_key,
                      allowed_domains, monthly_limit, current_month_usage,
                      plan, is_active, created_at
        """),
        {"id": key_id, "secret_key": new_secret},
    )
    await db.commit()
    return _to_out(dict(result.mappings().first()), mask_secret=False)


# ── 6. 삭제 ───────────────────────────────────────────────────────────────────

@router.delete("/keys/{key_id}", status_code=204)
async def delete_my_key(
    key_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    await _assert_owner(db, key_id, str(current_user.id))

    await db.execute(
        text("DELETE FROM saas_api_usage_logs WHERE api_key_id = :id"),
        {"id": key_id},
    )
    await db.execute(
        text("DELETE FROM saas_api_keys WHERE id = :id"),
        {"id": key_id},
    )
    await db.commit()


# ── 7. 사용 로그 ──────────────────────────────────────────────────────────────

@router.get("/keys/{key_id}/usage", response_model=UsageLogListResponse)
async def get_my_usage_logs(
    key_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
):
    await _assert_owner(db, key_id, str(current_user.id))

    total = (await db.execute(
        text("SELECT COUNT(*) FROM saas_api_usage_logs WHERE api_key_id = :id"),
        {"id": key_id},
    )).scalar() or 0

    result = await db.execute(
        text("""
            SELECT id, endpoint, status_code, response_time_ms, created_at
            FROM saas_api_usage_logs
            WHERE api_key_id = :id
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {"id": key_id, "limit": size, "offset": (page - 1) * size},
    )

    return UsageLogListResponse(
        total=total,
        items=[
            UsageLogOut(
                id=str(r["id"]),
                endpoint=r["endpoint"],
                status_code=r["status_code"],
                response_time_ms=r.get("response_time_ms"),
                created_at=_fmt(r.get("created_at")),
            )
            for r in result.mappings()
        ],
    )


# ── 8. 사용량 요약 ────────────────────────────────────────────────────────────

@router.get("/usage-summary", response_model=UsageSummaryOut)
async def get_my_usage_summary(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
    service_type: Optional[ServiceType] = Query(None),
):
    conditions = ["created_by = :user_id"]
    params: dict = {"user_id": str(current_user.id)}

    if service_type:
        conditions.append("service_type = :service_type")
        params["service_type"] = service_type

    where = " WHERE " + " AND ".join(conditions)
    result = (await db.execute(
        text(f"""
            SELECT COUNT(*) as total_keys,
                   COUNT(*) FILTER (WHERE is_active) as active_keys,
                   COALESCE(SUM(current_month_usage), 0) as total_usage
            FROM saas_api_keys{where}
        """),
        params,
    )).mappings().first()

    return UsageSummaryOut(
        total_keys=result["total_keys"] or 0,
        active_keys=result["active_keys"] or 0,
        total_usage_this_month=result["total_usage"] or 0,
    )


# ── 9. 플랜 문의 접수 ─────────────────────────────────────────────────────────

@router.post("/plan-inquiry", response_model=PlanInquiryOut, status_code=201)
async def create_plan_inquiry(
    payload: PlanInquiryCreateRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    # 동일 서비스 타입 pending 문의 중복 방지
    existing = (await db.execute(
        text("""
            SELECT id FROM saas_plan_inquiries
            WHERE user_id = :user_id
              AND service_type = :service_type
              AND status = 'pending'
            LIMIT 1
        """),
        {"user_id": str(current_user.id), "service_type": payload.service_type},
    )).mappings().first()

    if existing:
        raise HTTPException(status_code=400, detail="이미 대기 중인 문의가 있습니다.")

    result = await db.execute(
        text("""
            INSERT INTO saas_plan_inquiries
                (user_id, service_type, desired_plan, message, status)
            VALUES
                (:user_id, :service_type, :desired_plan, :message, 'pending')
            RETURNING id, user_id, service_type, desired_plan, message, status, created_at
        """),
        {
            "user_id": str(current_user.id),
            "service_type": payload.service_type,
            "desired_plan": payload.desired_plan,
            "message": payload.message,
        },
    )
    await db.commit()
    row = dict(result.mappings().first())

    return PlanInquiryOut(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        user_email=getattr(current_user, "email", None),
        service_type=row["service_type"],
        desired_plan=row["desired_plan"],
        message=row.get("message"),
        status=row["status"],
        created_at=_fmt(row.get("created_at")),
    )


# ── 10. 내 플랜 문의 목록 ─────────────────────────────────────────────────────

@router.get("/plan-inquiries", response_model=PlanInquiryListResponse)
async def get_my_plan_inquiries(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
    service_type: Optional[ServiceType] = Query(None),
):
    conditions = ["i.user_id = :user_id"]
    params: dict = {"user_id": str(current_user.id)}

    if service_type:
        conditions.append("i.service_type = :service_type")
        params["service_type"] = service_type

    where = " WHERE " + " AND ".join(conditions)

    total = (await db.execute(
        text(f"SELECT COUNT(*) FROM saas_plan_inquiries i{where}"), params
    )).scalar() or 0

    result = await db.execute(
        text(f"""
            SELECT i.id, i.user_id, u.email as user_email,
                   i.service_type, i.desired_plan, i.message,
                   i.status, i.created_at
            FROM saas_plan_inquiries i
            LEFT JOIN users u ON u.id = i.user_id
            {where}
            ORDER BY i.created_at DESC
        """),
        params,
    )

    return PlanInquiryListResponse(
        total=total,
        items=[
            PlanInquiryOut(
                id=str(r["id"]),
                user_id=str(r["user_id"]),
                user_email=r.get("user_email"),
                service_type=r["service_type"],
                desired_plan=r["desired_plan"],
                message=r.get("message"),
                status=r["status"],
                created_at=_fmt(r.get("created_at")),
            )
            for r in result.mappings()
        ],
    )
