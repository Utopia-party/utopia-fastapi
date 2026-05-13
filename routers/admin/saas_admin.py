"""
SaaS API 키 관리 — 파트너사 API 키 CRUD + 사용량 통계

관리자 페이지에서:
  - 파트너사별 API 키 발급/조회/수정/비활성화
  - 사용량 통계 대시보드
  - 사용 로그 조회
"""
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from .deps import require_admin_context, AdminContext

router = APIRouter(prefix="/admin/saas", tags=["admin-saas"])


# ══════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════

class ApiKeyCreateRequest(BaseModel):
    client_name: str = Field(..., min_length=1, max_length=200)
    allowed_domains: Optional[list[str]] = None
    monthly_limit: int = Field(default=1000, ge=100)
    plan: str = Field(default="free")


class ApiKeyUpdateRequest(BaseModel):
    client_name: Optional[str] = None
    allowed_domains: Optional[list[str]] = None
    monthly_limit: Optional[int] = Field(default=None, ge=100)
    plan: Optional[str] = None
    is_active: Optional[bool] = None


class ApiKeyOut(BaseModel):
    id: str
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


class ApiKeyListResponse(BaseModel):
    total: int
    items: list[ApiKeyOut]


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


class UsageStatsOut(BaseModel):
    total_keys: int
    active_keys: int
    total_usage_this_month: int
    top_clients: list[dict]


# ══════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════

def _generate_site_key() -> str:
    """pk_live_partyup_<16자리hex>"""
    return f"pk_live_partyup_{secrets.token_hex(16)}"


def _generate_secret_key() -> str:
    """sk_live_partyup_<32자리hex>"""
    return f"sk_live_partyup_{secrets.token_hex(32)}"


def _fmt_dt(value) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


# ══════════════════════════════════════════════════════════════
# 1. API 키 목록 조회
# ══════════════════════════════════════════════════════════════

@router.get("/keys", response_model=ApiKeyListResponse)
async def list_api_keys(
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
):
    """API 키 목록 (검색, 필터, 페이징)"""
    conditions = []
    params: dict = {}

    if search:
        conditions.append("(client_name ILIKE :search OR api_key ILIKE :search)")
        params["search"] = f"%{search}%"
    if is_active is not None:
        conditions.append("is_active = :is_active")
        params["is_active"] = is_active

    where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

    # 총 개수
    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM api_keys{where_clause}"), params
    )
    total = count_result.scalar() or 0

    # 데이터
    params["limit"] = size
    params["offset"] = (page - 1) * size
    result = await db.execute(
        text(f"""
            SELECT id, client_name, api_key, secret_key, allowed_domains,
                   monthly_limit, current_month_usage, plan, is_active,
                   created_at, updated_at
            FROM api_keys
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )

    items = []
    for row in result.mappings():
        items.append(ApiKeyOut(
            id=str(row["id"]),
            client_name=row["client_name"],
            api_key=row["api_key"],
            secret_key=row["secret_key"],
            allowed_domains=row["allowed_domains"],
            monthly_limit=row["monthly_limit"],
            current_month_usage=row["current_month_usage"],
            plan=row["plan"],
            is_active=row["is_active"],
            created_at=_fmt_dt(row.get("created_at")),
            updated_at=_fmt_dt(row.get("updated_at")),
        ))

    return ApiKeyListResponse(total=total, items=items)


# ══════════════════════════════════════════════════════════════
# 2. API 키 발급 (생성)
# ══════════════════════════════════════════════════════════════

@router.post("/keys", response_model=ApiKeyOut, status_code=201)
async def create_api_key(
    payload: ApiKeyCreateRequest,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    """새 파트너 API 키 발급"""
    site_key = _generate_site_key()
    secret_key = _generate_secret_key()

    result = await db.execute(
        text("""
            INSERT INTO api_keys (client_name, api_key, secret_key, allowed_domains,
                                  monthly_limit, plan, is_active, current_month_usage)
            VALUES (:client_name, :api_key, :secret_key, :allowed_domains,
                    :monthly_limit, :plan, true, 0)
            RETURNING id, client_name, api_key, secret_key, allowed_domains,
                      monthly_limit, current_month_usage, plan, is_active,
                      created_at, updated_at
        """),
        {
            "client_name": payload.client_name,
            "api_key": site_key,
            "secret_key": secret_key,
            "allowed_domains": payload.allowed_domains,
            "monthly_limit": payload.monthly_limit,
            "plan": payload.plan,
        },
    )
    await db.commit()
    row = result.mappings().first()

    return ApiKeyOut(
        id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=row["secret_key"],
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt_dt(row.get("created_at")),
        updated_at=_fmt_dt(row.get("updated_at")),
    )


# ══════════════════════════════════════════════════════════════
# 3. API 키 단건 조회
# ══════════════════════════════════════════════════════════════

@router.get("/keys/{key_id}", response_model=ApiKeyOut)
async def get_api_key(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    """API 키 상세 조회"""
    result = await db.execute(
        text("""
            SELECT id, client_name, api_key, secret_key, allowed_domains,
                   monthly_limit, current_month_usage, plan, is_active,
                   created_at, updated_at
            FROM api_keys WHERE id = :id
        """),
        {"id": key_id},
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    return ApiKeyOut(
        id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=row["secret_key"],
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt_dt(row.get("created_at")),
        updated_at=_fmt_dt(row.get("updated_at")),
    )


# ══════════════════════════════════════════════════════════════
# 4. API 키 수정
# ══════════════════════════════════════════════════════════════

@router.put("/keys/{key_id}", response_model=ApiKeyOut)
async def update_api_key(
    key_id: str,
    payload: ApiKeyUpdateRequest,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    """API 키 정보 수정 (이름, 도메인, 쿼터, 플랜, 활성상태)"""
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
        # 플랜 변경 시 monthly_limit 자동 연동 (명시적으로 monthly_limit를 함께 보내지 않은 경우)
        if payload.monthly_limit is None:
            plan_limits = {"free": 1000, "starter": 5000, "pro": 30000, "enterprise": 1000000}
            if payload.plan in plan_limits:
                set_parts.append("monthly_limit = :monthly_limit")
                params["monthly_limit"] = plan_limits[payload.plan]
    if payload.is_active is not None:
        set_parts.append("is_active = :is_active")
        params["is_active"] = payload.is_active

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
                      created_at, updated_at
        """),
        params,
    )
    await db.commit()
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    return ApiKeyOut(
        id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=row["secret_key"],
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt_dt(row.get("created_at")),
        updated_at=_fmt_dt(row.get("updated_at")),
    )


# ══════════════════════════════════════════════════════════════
# 5. API 키 재발급 (secret_key만 교체)
# ══════════════════════════════════════════════════════════════

@router.post("/keys/{key_id}/rotate-secret", response_model=ApiKeyOut)
async def rotate_secret_key(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    """secret_key 재발급 (site_key는 유지)"""
    new_secret = _generate_secret_key()
    result = await db.execute(
        text("""
            UPDATE api_keys
            SET secret_key = :secret_key, updated_at = NOW()
            WHERE id = :id
            RETURNING id, client_name, api_key, secret_key, allowed_domains,
                      monthly_limit, current_month_usage, plan, is_active,
                      created_at, updated_at
        """),
        {"id": key_id, "secret_key": new_secret},
    )
    await db.commit()
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    return ApiKeyOut(
        id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=row["secret_key"],
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        plan=row["plan"],
        is_active=row["is_active"],
        created_at=_fmt_dt(row.get("created_at")),
        updated_at=_fmt_dt(row.get("updated_at")),
    )


# ══════════════════════════════════════════════════════════════
# 6. 사용량 초기화 (월간 리셋)
# ══════════════════════════════════════════════════════════════

@router.post("/keys/{key_id}/reset-usage")
async def reset_monthly_usage(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    """월간 사용량 수동 초기화"""
    result = await db.execute(
        text("""
            UPDATE api_keys
            SET current_month_usage = 0, updated_at = NOW()
            WHERE id = :id
            RETURNING id
        """),
        {"id": key_id},
    )
    await db.commit()
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    return {"status": "ok", "message": "사용량이 초기화되었습니다."}


# ══════════════════════════════════════════════════════════════
# 6-1. API 키 삭제 (관리자 전용)
# ══════════════════════════════════════════════════════════════

@router.delete("/keys/{key_id}", status_code=204)
async def delete_api_key(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    """관리자 전용 API 키 삭제 — 관련 사용 로그도 함께 삭제"""
    # 키 존재 확인
    result = await db.execute(
        text("SELECT id FROM api_keys WHERE id = :id"),
        {"id": key_id},
    )
    if not result.mappings().first():
        raise HTTPException(status_code=404, detail="API 키를 찾을 수 없습니다.")

    # 관련 사용 로그 먼저 삭제
    await db.execute(
        text("DELETE FROM api_usage_logs WHERE api_key_id = :key_id"),
        {"key_id": key_id},
    )
    # API 키 삭제
    await db.execute(
        text("DELETE FROM api_keys WHERE id = :id"),
        {"id": key_id},
    )
    await db.commit()


# ══════════════════════════════════════════════════════════════
# 7. 사용 로그 조회
# ══════════════════════════════════════════════════════════════

@router.get("/keys/{key_id}/logs", response_model=UsageLogListResponse)
async def get_usage_logs(
    key_id: str,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
):
    """특정 API 키의 사용 로그"""
    count_result = await db.execute(
        text("SELECT COUNT(*) FROM api_usage_logs WHERE api_key_id = :key_id"),
        {"key_id": key_id},
    )
    total = count_result.scalar() or 0

    result = await db.execute(
        text("""
            SELECT id, endpoint, client_ip::text as client_ip, origin_domain,
                   status_code, response_time_ms, created_at
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
            client_ip=row.get("client_ip"),
            origin_domain=row.get("origin_domain"),
            status_code=row["status_code"],
            response_time_ms=row["response_time_ms"],
            created_at=_fmt_dt(row.get("created_at")),
        ))

    return UsageLogListResponse(total=total, items=items)


# ══════════════════════════════════════════════════════════════
# 8. SaaS 전체 통계
# ══════════════════════════════════════════════════════════════

@router.get("/stats", response_model=UsageStatsOut)
async def get_saas_stats(
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    """SaaS 대시보드 통계"""
    # 전체/활성 키 수
    key_counts = await db.execute(text("""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE is_active) as active
        FROM api_keys
    """))
    counts = key_counts.mappings().first()

    # 이번 달 전체 사용량
    usage_result = await db.execute(text("""
        SELECT COALESCE(SUM(current_month_usage), 0) as total_usage
        FROM api_keys
    """))
    total_usage = usage_result.scalar() or 0

    # 상위 사용 클라이언트
    top_result = await db.execute(text("""
        SELECT client_name, api_key, current_month_usage, monthly_limit, plan, is_active
        FROM api_keys
        ORDER BY current_month_usage DESC
        LIMIT 10
    """))
    top_clients = []
    for row in top_result.mappings():
        top_clients.append({
            "client_name": row["client_name"],
            "api_key": row["api_key"],
            "usage": row["current_month_usage"],
            "limit": row["monthly_limit"],
            "plan": row["plan"],
            "is_active": row["is_active"],
            "usage_percent": round(
                (row["current_month_usage"] / row["monthly_limit"] * 100)
                if row["monthly_limit"] > 0 else 0, 1
            ),
        })

    return UsageStatsOut(
        total_keys=counts["total"],
        active_keys=counts["active"],
        total_usage_this_month=total_usage,
        top_clients=top_clients,
    )


# ══════════════════════════════════════════════════════════════
# 플랜 문의 관리
# ══════════════════════════════════════════════════════════════

class PlanInquiryAdminOut(BaseModel):
    id: str
    user_id: str
    user_email: Optional[str] = None
    desired_plan: str
    message: Optional[str] = None
    status: str
    created_at: Optional[str] = None


class PlanInquiryListOut(BaseModel):
    total: int
    items: list[PlanInquiryAdminOut]


class PlanInquiryStatusUpdate(BaseModel):
    status: str = Field(..., description="변경할 상태: completed")


def _fmt_dt(value) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


@router.get(
    "/plan-inquiries",
    response_model=PlanInquiryListOut,
    summary="플랜 문의 목록 (관리자)",
)
async def list_plan_inquiries(
    status_filter: Optional[str] = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _admin: AdminContext = Depends(require_admin_context),
):
    """관리자가 전체 플랜 문의 목록을 조회한다."""
    where = ""
    params: dict = {}
    if status_filter:
        where = "WHERE pi.status = :status"
        params["status"] = status_filter

    count_result = await db.execute(
        text(f"SELECT COUNT(*) FROM plan_inquiries pi {where}"),
        params,
    )
    total = count_result.scalar() or 0

    params["limit"] = size
    params["offset"] = (page - 1) * size
    result = await db.execute(
        text(f"""
            SELECT pi.id, pi.user_id, u.email AS user_email,
                   pi.desired_plan, pi.message, pi.status, pi.created_at
            FROM plan_inquiries pi
            LEFT JOIN users u ON u.id = pi.user_id
            {where}
            ORDER BY pi.created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params,
    )
    rows = result.mappings().all()

    return PlanInquiryListOut(
        total=total,
        items=[
            PlanInquiryAdminOut(
                id=str(r["id"]),
                user_id=str(r["user_id"]),
                user_email=r["user_email"],
                desired_plan=r["desired_plan"],
                message=r["message"],
                status=r["status"],
                created_at=_fmt_dt(r["created_at"]),
            )
            for r in rows
        ],
    )


@router.put(
    "/plan-inquiries/{inquiry_id}",
    response_model=PlanInquiryAdminOut,
    summary="플랜 문의 상태 변경 (관리자)",
)
async def update_plan_inquiry_status(
    inquiry_id: str,
    payload: PlanInquiryStatusUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: AdminContext = Depends(require_admin_context),
):
    """관리자가 문의 상태를 변경한다 (pending → completed)."""
    if payload.status not in ("pending", "completed"):
        raise HTTPException(status_code=400, detail="유효한 상태: pending, completed")

    result = await db.execute(
        text("""
            UPDATE plan_inquiries SET status = :status
            WHERE id = :id
            RETURNING id, user_id, desired_plan, message, status, created_at
        """),
        {"id": inquiry_id, "status": payload.status},
    )
    await db.commit()
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="문의를 찾을 수 없습니다.")

    # 이메일 조회
    user_result = await db.execute(
        text("SELECT email FROM users WHERE id = :uid"),
        {"uid": str(row["user_id"])},
    )
    user_row = user_result.mappings().first()

    return PlanInquiryAdminOut(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        user_email=user_row["email"] if user_row else None,
        desired_plan=row["desired_plan"],
        message=row["message"],
        status=row["status"],
        created_at=_fmt_dt(row["created_at"]),
    )
