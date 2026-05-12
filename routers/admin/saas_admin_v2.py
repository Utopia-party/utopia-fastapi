import secrets
from datetime import datetime
from typing import Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from routers.admin.deps import require_admin_context, AdminContext

router = APIRouter(prefix="/admin/saas-v2", tags=["admin-saas-v2"])

ServiceType = Literal["captcha_l2", "chat_filter"]


# ── Schemas ──────────────────────────────────────────────────────────────────

class KeyCreateRequest(BaseModel):
    service_type: ServiceType
    client_name: str = Field(..., min_length=1, max_length=200)
    allowed_domains: Optional[list[str]] = None
    monthly_limit: int = Field(default=10000, ge=100)
    plan: str = Field(default="free")


class KeyUpdateRequest(BaseModel):
    client_name: Optional[str] = None
    allowed_domains: Optional[list[str]] = None
    monthly_limit: Optional[int] = Field(default=None, ge=100)
    plan: Optional[str] = None
    is_active: Optional[bool] = None


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
    updated_at: Optional[str]


class KeyListResponse(BaseModel):
    total: int
    items: list[KeyOut]


class UsageLogOut(BaseModel):
    id: str
    endpoint: str
    client_ip: Optional[str]
    origin_domain: Optional[str]
    status_code: int
    response_time_ms: int
    created_at: Optional[str]


class UsageLogListResponse(BaseModel):
    total: int
    items: list[UsageLogOut]


class StatsOut(BaseModel):
    total_keys: int
    active_keys: int
    total_usage_this_month: int
    top_clients: list[dict]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _gen_key(service_type: str) -> tuple[str, str]:
    prefix = "l2" if service_type == "captcha_l2" else "chat"
    api_key = f"pk_{prefix}_partyup_{secrets.token_hex(16)}"
    secret_key = f"sk_{prefix}_partyup_{secrets.token_hex(32)}"
    return api_key, secret_key


def _fmt(value) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _to_out(row: dict) -> KeyOut:
    return KeyOut(
        id=str(row["id"]),
        service_type=row["service_type"],
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=row["secret_key"],
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt(row.get("created_at")),
        updated_at=_fmt(row.get("updated_at")),
    )


# ── 1. 목록 조회 ──────────────────────────────────────────────────────────────

@router.get("/keys", response_model=KeyListResponse)
async def list_keys(
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    service_type: Optional[ServiceType] = Query(None),
    search: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
):
    conditions = []
    params: dict = {}

    if service_type:
        conditions.append("service_type = :service_type")
        params["service_type"] = service_type
    if search:
        conditions.append("(client_name ILIKE :search OR api_key ILIKE :search)")
        params["search"] = f"%{search}%"
    if is_active is not None:
        conditions.append("is_active = :is_active")
        params["is_active"] = is_active

    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM saas_api_keys{where}"), params
    )
    total = count_result.scalar() or 0

    params["limit"] = size
    params["offset"] = (page - 1) * size
    result = await db.execute(
        text(f"""
            SELECT id, service_type, client_name, api_key, secret_key,
                   allowed_domains, monthly_limit, current_month_usage,
                   plan, is_active, created_at, updated_at
            FROM saas_api_keys
            {where}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )

    return KeyListResponse(
        total=total,
        items=[_to_out(dict(r)) for r in result.mappings()],
    )


# ── 2. 발급 ───────────────────────────────────────────────────────────────────

@router.post("/keys", response_model=KeyOut, status_code=201)
async def create_key(
    payload: KeyCreateRequest,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    api_key, secret_key = _gen_key(payload.service_type)

    result = await db.execute(
        text("""
            INSERT INTO saas_api_keys
                (service_type, client_name, api_key, secret_key,
                 allowed_domains, monthly_limit, plan, is_active, current_month_usage)
            VALUES
                (:service_type, :client_name, :api_key, :secret_key,
                 :allowed_domains, :monthly_limit, :plan, true, 0)
            RETURNING id, service_type, client_name, api_key, secret_key,
                      allowed_domains, monthly_limit, current_month_usage,
                      plan, is_active, created_at, updated_at
        """),
        {
            "service_type": payload.service_type,
            "client_name": payload.client_name,
            "api_key": api_key,
            "secret_key": secret_key,
            "allowed_domains": payload.allowed_domains,
            "monthly_limit": payload.monthly_limit,
            "plan": payload.plan,
        },
    )
    await db.commit()
    return _to_out(dict(result.mappings().first()))


# ── 3. 단건 조회 ──────────────────────────────────────────────────────────────

@router.get("/keys/{key_id}", response_model=KeyOut)
async def get_key(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        text("""
            SELECT id, service_type, client_name, api_key, secret_key,
                   allowed_domains, monthly_limit, current_month_usage,
                   plan, is_active, created_at, updated_at
            FROM saas_api_keys WHERE id = :id
        """),
        {"id": key_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="키를 찾을 수 없습니다.")
    return _to_out(dict(row))


# ── 4. 수정 ───────────────────────────────────────────────────────────────────

@router.put("/keys/{key_id}", response_model=KeyOut)
async def update_key(
    key_id: str,
    payload: KeyUpdateRequest,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    set_parts = []
    params: dict = {"id": key_id}

    if payload.client_name is not None:
        set_parts.append("client_name = :client_name")
        params["client_name"] = payload.client_name
    if payload.allowed_domains is not None:
        set_parts.append("allowed_domains = :allowed_domains")
        params["allowed_domains"] = payload.allowed_domains
    if payload.monthly_limit is not None:
        set_parts.append("monthly_limit = :monthly_limit")
        params["monthly_limit"] = payload.monthly_limit
    if payload.plan is not None:
        set_parts.append("plan = :plan")
        params["plan"] = payload.plan
    if payload.is_active is not None:
        set_parts.append("is_active = :is_active")
        params["is_active"] = payload.is_active

    if not set_parts:
        raise HTTPException(status_code=400, detail="변경할 항목이 없습니다.")

    set_parts.append("updated_at = NOW()")
    result = await db.execute(
        text(f"""
            UPDATE saas_api_keys SET {', '.join(set_parts)}
            WHERE id = :id
            RETURNING id, service_type, client_name, api_key, secret_key,
                      allowed_domains, monthly_limit, current_month_usage,
                      plan, is_active, created_at, updated_at
        """),
        params,
    )
    await db.commit()
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="키를 찾을 수 없습니다.")
    return _to_out(dict(row))


# ── 5. Secret 재발급 ──────────────────────────────────────────────────────────

@router.post("/keys/{key_id}/rotate-secret", response_model=KeyOut)
async def rotate_secret(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    # service_type 먼저 조회 (prefix 유지)
    svc_result = await db.execute(
        text("SELECT service_type FROM saas_api_keys WHERE id = :id"),
        {"id": key_id},
    )
    svc_row = svc_result.mappings().first()
    if not svc_row:
        raise HTTPException(status_code=404, detail="키를 찾을 수 없습니다.")

    _, new_secret = _gen_key(svc_row["service_type"])
    result = await db.execute(
        text("""
            UPDATE saas_api_keys
            SET secret_key = :secret_key, updated_at = NOW()
            WHERE id = :id
            RETURNING id, service_type, client_name, api_key, secret_key,
                      allowed_domains, monthly_limit, current_month_usage,
                      plan, is_active, created_at, updated_at
        """),
        {"id": key_id, "secret_key": new_secret},
    )
    await db.commit()
    return _to_out(dict(result.mappings().first()))


# ── 6. 사용량 초기화 ──────────────────────────────────────────────────────────

@router.post("/keys/{key_id}/reset-usage")
async def reset_usage(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        text("UPDATE saas_api_keys SET current_month_usage = 0, updated_at = NOW() WHERE id = :id RETURNING id"),
        {"id": key_id},
    )
    await db.commit()
    if not result.mappings().first():
        raise HTTPException(status_code=404, detail="키를 찾을 수 없습니다.")
    return {"status": "ok"}


# ── 7. 사용 로그 ──────────────────────────────────────────────────────────────

@router.get("/keys/{key_id}/logs", response_model=UsageLogListResponse)
async def get_logs(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
):
    total = (await db.execute(
        text("SELECT COUNT(*) FROM saas_api_usage_logs WHERE api_key_id = :id"),
        {"id": key_id},
    )).scalar() or 0

    result = await db.execute(
        text("""
            SELECT id, endpoint, client_ip::text, origin_domain,
                   status_code, response_time_ms, created_at
            FROM saas_api_usage_logs
            WHERE api_key_id = :id
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {"id": key_id, "limit": size, "offset": (page - 1) * size},
    )

    items = [
        UsageLogOut(
            id=str(r["id"]),
            endpoint=r["endpoint"],
            client_ip=r.get("client_ip"),
            origin_domain=r.get("origin_domain"),
            status_code=r["status_code"],
            response_time_ms=r["response_time_ms"],
            created_at=_fmt(r.get("created_at")),
        )
        for r in result.mappings()
    ]
    return UsageLogListResponse(total=total, items=items)


# ── 8. 전체 통계 ──────────────────────────────────────────────────────────────

@router.get("/stats", response_model=StatsOut)
async def get_stats(
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
    service_type: Optional[ServiceType] = Query(None),
):
    where = "WHERE service_type = :st" if service_type else ""
    params = {"st": service_type} if service_type else {}

    counts = (await db.execute(
        text(f"""
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE is_active) as active
            FROM saas_api_keys {where}
        """), params
    )).mappings().first()

    total_usage = (await db.execute(
        text(f"SELECT COALESCE(SUM(current_month_usage), 0) FROM saas_api_keys {where}"),
        params,
    )).scalar() or 0

    top_result = await db.execute(
        text(f"""
            SELECT client_name, api_key, service_type,
                   current_month_usage, monthly_limit, plan, is_active
            FROM saas_api_keys {where}
            ORDER BY current_month_usage DESC
            LIMIT 10
        """), params
    )
    top_clients = [
        {
            "client_name": r["client_name"],
            "api_key": r["api_key"],
            "service_type": r["service_type"],
            "usage": r["current_month_usage"],
            "limit": r["monthly_limit"],
            "plan": r["plan"],
            "is_active": r["is_active"],
            "usage_percent": round(
                (r["current_month_usage"] / r["monthly_limit"] * 100)
                if r["monthly_limit"] > 0 else 0, 1
            ),
        }
        for r in top_result.mappings()
    ]

    return StatsOut(
        total_keys=counts["total"],
        active_keys=counts["active"],
        total_usage_this_month=total_usage,
        top_clients=top_clients,
    )


# ── Plan Inquiry ──────────────────────────────────────────────────────────────

class PlanInquiryOut(BaseModel):
    id: str
    user_id: str
    user_email: Optional[str]
    service_type: str
    desired_plan: str
    message: Optional[str]
    status: str
    created_at: Optional[str]


class PlanInquiryListOut(BaseModel):
    total: int
    items: list[PlanInquiryOut]


class PlanInquiryUpdateRequest(BaseModel):
    status: str


@router.get("/plan-inquiries", response_model=PlanInquiryListOut)
async def list_v2_plan_inquiries(
    service_type: Optional[ServiceType] = Query(None),
    status: Optional[str] = Query(None),
    size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    _: AdminContext = Depends(require_admin_context),
):
    conditions = []
    params: dict = {}

    if service_type:
        conditions.append("i.service_type = :service_type")
        params["service_type"] = service_type
    if status:
        conditions.append("i.status = :status")
        params["status"] = status

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM saas_plan_inquiries i {where}"), params
    )
    total = count_result.scalar() or 0

    params["size"] = size
    rows = (await db.execute(
        text(f"""
            SELECT i.id, i.user_id, u.email as user_email,
                   i.service_type, i.desired_plan, i.message,
                   i.status, i.created_at::text
            FROM saas_plan_inquiries i
            LEFT JOIN users u ON u.id = i.user_id
            {where}
            ORDER BY i.created_at DESC
            LIMIT :size
        """), params
    )).mappings().all()

    return PlanInquiryListOut(
        total=total,
        items=[
            PlanInquiryOut(
                id=str(r["id"]),
                user_id=str(r["user_id"]),
                user_email=r["user_email"],
                service_type=r["service_type"],
                desired_plan=r["desired_plan"],
                message=r["message"],
                status=r["status"],
                created_at=r["created_at"],
            )
            for r in rows
        ],
    )


@router.put("/plan-inquiries/{inquiry_id}", response_model=PlanInquiryOut)
async def update_v2_plan_inquiry_status(
    inquiry_id: str,
    body: PlanInquiryUpdateRequest,
    db: AsyncSession = Depends(get_db),
    _: AdminContext = Depends(require_admin_context),
):
    result = await db.execute(
        text("""
            UPDATE saas_plan_inquiries
            SET status = :status
            WHERE id = :id
            RETURNING id, user_id, service_type, desired_plan, message, status, created_at::text
        """),
        {"id": inquiry_id, "status": body.status},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="문의를 찾을 수 없습니다.")
    await db.commit()

    return PlanInquiryOut(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        user_email=None,
        service_type=row["service_type"],
        desired_plan=row["desired_plan"],
        message=row["message"],
        status=row["status"],
        created_at=row["created_at"],
    )
