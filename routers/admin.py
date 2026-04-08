from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from core.database import get_db
from core.minio_assets import build_minio_asset_url
from core.security import require_user
from models.admin import (
    ActivityLog,
    AdminRole,
    ModerationAction,
    Receipt,
    Report,
    Settlement,
    SystemLog,
)
from models.notification import Notification
from models.party import Party, PartyMember, Service
from models.user import User
from schemas.admin import (
    AdminDashboardOut,
    AdminPartyActionIn,
    AdminPartyRecordOut,
    AdminPermissionOut,
    AdminRoleRecordOut,
    AdminRoleUpdateIn,
    AdminServiceRecordOut,
    AdminServiceUpdateIn,
    AdminStatusUpdateIn,
    AdminUserDetailOut,
    AdminUserRecordOut,
    AdminUserStatusUpdateIn,
    ReceiptRecordOut,
    ReportRecordOut,
    SettlementRecordOut,
    SystemLogRecordOut,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@dataclass
class AdminContext:
    user: User
    role: AdminRole


def _format_datetime(value: datetime | None) -> str:
    if not value:
        return "-"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _format_relative(value: datetime | None) -> str:
    if not value:
        return "-"

    delta = datetime.now(timezone.utc) - value.astimezone(timezone.utc)
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "방금"
    if seconds < 3600:
        return f"{seconds // 60}분 전"
    if seconds < 86400:
        return f"{seconds // 3600}시간 전"
    return f"{seconds // 86400}일 전"


def _to_int(value: Decimal | int | float | None) -> int:
    if value is None:
        return 0
    if isinstance(value, Decimal):
        return int(float(value))
    return int(value)


def _admin_permissions_for_role(role: str) -> dict[str, Any]:
    role = role.upper()
    if role == "ROOT":
        return {
            "can_manage_users": True,
            "can_manage_parties": True,
            "can_manage_reports": True,
            "can_manage_moderation": True,
            "can_approve_receipts": True,
            "can_approve_settlements": True,
            "can_view_logs": True,
            "can_manage_admins": True,
        }
    return {
        "can_manage_users": True,
        "can_manage_parties": True,
        "can_manage_reports": True,
        "can_manage_moderation": True,
        "can_approve_receipts": True,
        "can_approve_settlements": True,
        "can_view_logs": True,
        "can_manage_admins": False,
    }


def _manual_status_label(action_type: str | None) -> str | None:
    if not action_type:
        return None
    return {
        "STATUS_정상": "정상",
        "STATUS_주의": "주의",
        "STATUS_정지": "정지",
    }.get(action_type.upper())


def _user_status_label(user: User, report_count: int, manual_status: str | None = None) -> str:
    if not user.is_active:
        return "정지"
    if manual_status in {"정상", "주의"}:
        return manual_status
    if _to_int(user.trust_score) < 75 or report_count >= 2:
        return "주의"
    return "정상"


def _party_status_label(party: Party, report_count: int) -> str:
    if party.status.lower() == "ended":
        return "종료 예정"
    if report_count > 0:
        return "위험"
    if party.status.lower() == "recruiting":
        return "모집중"
    return "운영중"


def _report_status_label(value: str) -> str:
    return {
        "pending": "접수",
        "processed": "처리",
        "rejected": "기각",
        "appealed": "이의제기",
        "auto_processed": "AI처리",
    }.get(value.lower(), value)


def _report_status_code(value: str) -> str:
    return {
        "접수": "pending",
        "처리": "processed",
        "기각": "rejected",
        "이의제기": "appealed",
        "AI처리": "auto_processed",
    }.get(value, value.lower())


def _report_type_label(value: str) -> str:
    return {
        "user": "사용자",
        "party": "파티",
        "chat": "채팅",
    }.get(value.lower(), value)


def _report_target_counts_subquery(target_type: str, label: str):
    return (
        select(Report.target_id.label(label), func.count(Report.id).label("count"))
        .where(func.lower(Report.target_type) == target_type)
        .group_by(Report.target_id)
        .subquery()
    )


def _receipt_status_label(value: str) -> str:
    return {
        "PENDING": "대기",
        "APPROVED": "승인",
        "REJECTED": "거절",
    }.get(value.upper(), value)


def _receipt_status_code(value: str) -> str:
    return {
        "대기": "PENDING",
        "승인": "APPROVED",
        "거절": "REJECTED",
    }.get(value, value.upper())


def _settlement_status_label(value: str) -> str:
    return {
        "PENDING": "대기",
        "APPROVED": "승인",
        "REJECTED": "거절",
    }.get(value.upper(), value)


def _settlement_status_code(value: str) -> str:
    return {
        "대기": "PENDING",
        "승인": "APPROVED",
        "거절": "REJECTED",
    }.get(value, value.upper())


async def _append_activity_log(
    db: AsyncSession,
    *,
    actor_user_id: Any | None,
    action_type: str,
    description: str,
    path: str | None = None,
    ip_address: str | None = None,
) -> None:
    db.add(
        ActivityLog(
            actor_user_id=actor_user_id,
            action_type=action_type,
            description=description,
            path=path,
            ip_address=ip_address,
        )
    )


async def _append_system_log(
    db: AsyncSession,
    *,
    level: str,
    service: str,
    message: str,
    actor: str | None = None,
    ) -> None:
    db.add(SystemLog(level=level, service=service, message=message, actor=actor))


def _admin_permissions_payload(payload: AdminRoleUpdateIn) -> dict[str, bool]:
    return {
        "can_manage_users": payload.canManageUsers,
        "can_manage_parties": payload.canManageParties,
        "can_manage_reports": payload.canManageReports,
        "can_manage_moderation": payload.canManageModeration,
        "can_approve_receipts": payload.canApproveReceipts,
        "can_approve_settlements": payload.canApproveSettlements,
        "can_view_logs": payload.canViewLogs,
        "can_manage_admins": payload.canManageAdmins,
    }


def _has_any_admin_permission(values: dict[str, bool]) -> bool:
    return any(values.values())


def _serialize_admin_permissions(role: AdminRole) -> AdminPermissionOut:
    return AdminPermissionOut(
        canManageUsers=role.can_manage_users,
        canManageParties=role.can_manage_parties,
        canManageReports=role.can_manage_reports,
        canManageModeration=role.can_manage_moderation,
        canApproveReceipts=role.can_approve_receipts,
        canApproveSettlements=role.can_approve_settlements,
        canViewLogs=role.can_view_logs,
        canManageAdmins=role.can_manage_admins,
    )


def _serialize_admin_role(role: AdminRole, user: User, created_by: User | None) -> AdminRoleRecordOut:
    return AdminRoleRecordOut(
        id=str(role.id),
        userId=str(user.id),
        adminId=user.nickname or user.email,
        canManageUsers=role.can_manage_users,
        canManageParties=role.can_manage_parties,
        canManageReports=role.can_manage_reports,
        canManageModeration=role.can_manage_moderation,
        canApproveReceipts=role.can_approve_receipts,
        canApproveSettlements=role.can_approve_settlements,
        canViewLogs=role.can_view_logs,
        canManageAdmins=role.can_manage_admins,
        lastUpdated=_format_datetime(role.updated_at),
        updatedBy=(created_by.nickname or created_by.email) if created_by else "system",
    )


def _serialize_admin_service(service: Service, created_by: User | None) -> AdminServiceRecordOut:
    return AdminServiceRecordOut(
        id=str(service.id),
        name=service.name,
        category=service.category,
        maxMembers=service.max_members,
        monthlyPrice=service.monthly_price,
        logoImageKey=service.logo_image_key,
        logoImageUrl=build_minio_asset_url(service.logo_image_key),
        isActive=service.is_active,
        createdBy=(created_by.nickname or created_by.email) if created_by else "-",
        createdAt=_format_datetime(service.created_at),
        updatedAt=_format_datetime(service.updated_at),
        commissionRate=float(service.commission_rate or 0),
        leaderDiscountRate=float(service.leader_discount_rate or 0),
        referralDiscountRate=float(service.referral_discount_rate or 0),
    )


def _assert_admin_permission(
    admin: AdminContext,
    permission_name: str,
    detail: str,
) -> AdminContext:
    if not getattr(admin.role, permission_name):
        raise HTTPException(status_code=403, detail=detail)
    return admin


def _latest_user_status_actions_subquery():
    ranked_actions = (
        select(
            ModerationAction.user_id.label("user_id"),
            ModerationAction.action_type.label("action_type"),
            func.row_number()
            .over(
                partition_by=ModerationAction.user_id,
                order_by=ModerationAction.created_at.desc(),
            )
            .label("row_num"),
        )
        .where(
            ModerationAction.user_id.is_not(None),
            ModerationAction.action_type.in_(["STATUS_정상", "STATUS_주의", "STATUS_정지"]),
        )
        .subquery()
    )
    return (
        select(ranked_actions.c.user_id, ranked_actions.c.action_type)
        .where(ranked_actions.c.row_num == 1)
        .subquery()
    )


async def _count_root_admins(db: AsyncSession) -> int:
    return int(
        await db.scalar(
            select(func.count()).select_from(AdminRole).where(AdminRole.can_manage_admins.is_(True))
        )
        or 0
    )


async def _ensure_admin_role(db: AsyncSession, user: User) -> AdminRole:
    result = await db.execute(select(AdminRole).where(AdminRole.user_id == user.id))
    role = result.scalar_one_or_none()
    if role:
        if role.created_by is None:
            role.created_by = user.id
            await db.commit()
            await db.refresh(role)
        return role

    existing_count = await db.scalar(select(func.count()).select_from(AdminRole)) or 0
    defaults = _admin_permissions_for_role("ROOT" if existing_count == 0 else "ADMIN")
    role = AdminRole(
        user_id=user.id,
        created_by=user.id,
        **defaults,
    )
    db.add(role)
    await db.commit()
    await db.refresh(role)
    return role


async def require_admin_context(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
) -> AdminContext:
    if (current_user.role or "").lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자만 접근할 수 있습니다.",
        )
    role = await _ensure_admin_role(db, current_user)
    return AdminContext(user=current_user, role=role)


async def require_admin_user_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_users", "사용자 관리 권한이 없습니다.")


async def require_admin_party_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_parties", "파티 관리 권한이 없습니다.")


async def require_admin_report_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_reports", "신고 관리 권한이 없습니다.")


async def require_admin_receipt_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_approve_receipts", "영수증 승인 권한이 없습니다.")


async def require_admin_settlement_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_approve_settlements", "정산 승인 권한이 없습니다.")


async def require_admin_log_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_view_logs", "시스템 로그 조회 권한이 없습니다.")


async def require_admin_role_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_admins", "관리자 권한 변경 권한이 없습니다.")


@router.get("/dashboard", response_model=AdminDashboardOut)
async def get_admin_dashboard(
    _: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    total_users = await db.scalar(select(func.count()).select_from(User)) or 0
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_signups = await db.scalar(
        select(func.count()).select_from(User).where(User.created_at >= today_start)
    ) or 0
    pending_reports = await db.scalar(
        select(func.count()).select_from(Report).where(func.lower(Report.status) == "pending")
    ) or 0
    pending_settlements = await db.scalar(
        select(func.count()).select_from(Settlement).where(Settlement.status == "PENDING")
    ) or 0

    active_users = await db.scalar(
        select(func.count()).select_from(User).where(User.is_active.is_(True))
    ) or 0
    suspended_users = await db.scalar(
        select(func.count()).select_from(User).where(User.is_active.is_(False))
    ) or 0
    admin_users = await db.scalar(
        select(func.count()).select_from(User).where(func.lower(User.role) == "admin")
    ) or 0

    approved_amount = await db.scalar(
        select(func.coalesce(func.sum(Receipt.ocr_amount), 0)).where(Receipt.status == "APPROVED")
    ) or 0
    pending_amount = await db.scalar(
        select(func.coalesce(func.sum(Receipt.ocr_amount), 0)).where(Receipt.status == "PENDING")
    ) or 0
    rejected_amount = await db.scalar(
        select(func.coalesce(func.sum(Receipt.ocr_amount), 0)).where(Receipt.status == "REJECTED")
    ) or 0

    return AdminDashboardOut(
        metrics=[
            {
                "id": "members",
                "label": "회원 수",
                "value": f"{total_users:,}",
                "helper": "현재 운영 중인 전체 회원 기준",
            },
            {
                "id": "today",
                "label": "오늘 가입",
                "value": f"+{today_signups}",
                "helper": "오늘 00:00 이후 가입 완료",
            },
            {
                "id": "reports",
                "label": "신고(접수)",
                "value": f"{pending_reports}",
                "helper": "실시간 검토 대기 건수",
            },
            {
                "id": "settlements",
                "label": "정산 승인 대기",
                "value": f"{pending_settlements}",
                "helper": "관리자 승인 대기 상태",
            },
        ],
        member_stats=[
            {"label": "활성 사용자(가입)", "value": f"{active_users}"},
            {"label": "정지 사용자", "value": f"{suspended_users}"},
            {"label": "관리자 계정", "value": f"{admin_users}"},
        ],
        sales_stats=[
            {"label": "이번달 승인 금액", "value": f"₩ {int(approved_amount):,}"},
            {"label": "대기 금액", "value": f"₩ {int(pending_amount):,}"},
            {"label": "거절 금액", "value": f"₩ {int(rejected_amount):,}"},
        ],
        today_summary=f"접수 신고 {pending_reports}건 / 정산 대기 {pending_settlements}건 / 실시간 가입 +{today_signups}",
    )


@router.get("/me", response_model=AdminPermissionOut)
async def get_admin_me(
    admin: AdminContext = Depends(require_admin_context),
):
    return _serialize_admin_permissions(admin.role)


@router.get("/roles", response_model=list[AdminRoleRecordOut])
async def get_admin_roles(
    _: AdminContext = Depends(require_admin_role_permission),
    db: AsyncSession = Depends(get_db),
):
    creator = aliased(User)
    result = await db.execute(
        select(AdminRole, User, creator)
        .join(User, AdminRole.user_id == User.id)
        .outerjoin(creator, AdminRole.created_by == creator.id)
        .order_by(AdminRole.updated_at.desc())
    )
    rows = result.all()

    return [_serialize_admin_role(role, user, created_by) for role, user, created_by in rows]


@router.put("/roles/{user_id}", response_model=AdminRoleRecordOut)
async def update_admin_role(
    user_id: str,
    payload: AdminRoleUpdateIn,
    admin: AdminContext = Depends(require_admin_role_permission),
    db: AsyncSession = Depends(get_db),
):
    target_user = await db.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="관리자 대상으로 지정한 사용자가 없습니다.")

    next_permissions = _admin_permissions_payload(payload)
    if not _has_any_admin_permission(next_permissions):
        raise HTTPException(status_code=400, detail="최소 하나 이상의 관리자 권한이 필요합니다.")

    result = await db.execute(select(AdminRole).where(AdminRole.user_id == target_user.id))
    role_row = result.scalar_one_or_none()

    if target_user.id == admin.user.id:
        raise HTTPException(
            status_code=400,
            detail="본인 관리자 권한은 직접 변경할 수 없습니다.",
        )

    if role_row and role_row.can_manage_admins and not next_permissions["can_manage_admins"]:
        if await _count_root_admins(db) <= 1:
            raise HTTPException(
                status_code=400,
                detail="마지막 ROOT 관리자는 권한을 변경할 수 없습니다.",
            )

    target_user.role = "admin"
    if role_row is None:
        role_row = AdminRole(
            user_id=target_user.id,
            created_by=admin.user.id,
            **next_permissions,
        )
        db.add(role_row)
    else:
        for key, value in next_permissions.items():
            setattr(role_row, key, value)
        if role_row.created_by is None:
            role_row.created_by = admin.user.id

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="admin_role_updated",
        description=f"{target_user.nickname} 관리자 권한 세트를 변경",
        path=f"/api/admin/roles/{user_id}",
    )
    await _append_system_log(
        db,
        level="INFO",
        service="admin",
        message=f"관리자 권한 변경: {target_user.nickname}",
        actor=admin.user.nickname,
    )
    await db.commit()
    await db.refresh(role_row)

    return _serialize_admin_role(role_row, target_user, admin.user)


@router.get("/users", response_model=list[AdminUserRecordOut])
async def get_admin_users(
    _: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    status_filter: str = Query(default="", alias="status"),
):
    report_counts = _report_target_counts_subquery("user", "user_id")
    party_counts = (
        select(PartyMember.user_id.label("user_id"), func.count(PartyMember.id).label("count"))
        .group_by(PartyMember.user_id)
        .subquery()
    )
    latest_status_actions = _latest_user_status_actions_subquery()

    stmt = (
        select(
            User,
            func.coalesce(report_counts.c.count, 0),
            func.coalesce(party_counts.c.count, 0),
            latest_status_actions.c.action_type,
        )
        .outerjoin(report_counts, report_counts.c.user_id == User.id)
        .outerjoin(party_counts, party_counts.c.user_id == User.id)
        .outerjoin(latest_status_actions, latest_status_actions.c.user_id == User.id)
        .order_by(User.created_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.all()

    items: list[AdminUserRecordOut] = []
    q = keyword.lower().strip()
    for user, report_count, party_count, manual_action in rows:
        status_label = _user_status_label(
            user,
            int(report_count),
            _manual_status_label(manual_action),
        )
        if status_filter and status_label != status_filter:
            continue
        if q and not (
            q in str(user.id).lower()
            or q in (user.nickname or "").lower()
            or q in status_label.lower()
        ):
            continue

        items.append(
            AdminUserRecordOut(
                id=str(user.id),
                nickname=user.nickname,
                status=status_label,
                reportCount=int(report_count),
                partyCount=int(party_count),
                trustScore=_to_int(user.trust_score),
                lastActive=_format_relative(user.last_login_at or user.updated_at),
            )
        )

    return items


@router.get("/users/{user_id}", response_model=AdminUserDetailOut)
async def get_admin_user_detail(
    user_id: str,
    _: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    report_count = await db.scalar(
        select(func.count())
        .select_from(Report)
        .where(func.lower(Report.target_type) == "user", Report.target_id == user.id)
    ) or 0
    party_count = await db.scalar(
        select(func.count()).select_from(PartyMember).where(PartyMember.user_id == user.id)
    ) or 0
    manual_action = await db.scalar(
        select(ModerationAction.action_type)
        .where(
            ModerationAction.user_id == user.id,
            ModerationAction.action_type.in_(["STATUS_정상", "STATUS_주의", "STATUS_정지"]),
        )
        .order_by(ModerationAction.created_at.desc())
        .limit(1)
    )

    return AdminUserDetailOut(
        id=str(user.id),
        email=user.email,
        nickname=user.nickname,
        name=user.name,
        phone=user.phone,
        role=user.role,
        status=_user_status_label(user, int(report_count), _manual_status_label(manual_action)),
        trustScore=_to_int(user.trust_score),
        reportCount=int(report_count),
        partyCount=int(party_count),
        createdAt=_format_datetime(user.created_at),
        lastActive=_format_datetime(user.last_login_at or user.updated_at),
    )


@router.get("/services", response_model=list[AdminServiceRecordOut])
async def get_admin_services(
    _: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
):
    creator = aliased(User)
    result = await db.execute(
        select(Service, creator)
        .outerjoin(creator, Service.created_by == creator.id)
        .order_by(Service.updated_at.desc(), Service.created_at.desc())
    )
    rows = result.all()
    return [_serialize_admin_service(service, created_by) for service, created_by in rows]


@router.patch("/services/{service_id}", response_model=AdminServiceRecordOut)
async def update_admin_service(
    service_id: str,
    payload: AdminServiceUpdateIn,
    admin: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
):
    service = await db.get(Service, service_id)
    if not service:
        raise HTTPException(status_code=404, detail="서비스를 찾을 수 없습니다.")

    service.max_members = payload.maxMembers
    service.monthly_price = payload.monthlyPrice
    service.logo_image_key = payload.logoImageKey
    service.is_active = payload.isActive
    service.commission_rate = payload.commissionRate
    service.leader_discount_rate = payload.leaderDiscountRate
    service.referral_discount_rate = payload.referralDiscountRate

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="admin_service_updated",
        description=f"{service.name} 서비스 운영 값을 수정",
        path=f"/api/admin/services/{service_id}",
    )
    await _append_system_log(
        db,
        level="INFO",
        service="admin",
        message=f"서비스 설정 변경: {service.name}",
        actor=admin.user.nickname,
    )
    await db.commit()
    await db.refresh(service)

    created_by = await db.get(User, service.created_by) if service.created_by else None
    return _serialize_admin_service(service, created_by)


@router.patch("/users/{user_id}/status", response_model=AdminUserRecordOut)
async def update_admin_user_status(
    user_id: str,
    payload: AdminUserStatusUpdateIn,
    admin: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
):
    target_user = await db.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    if payload.status not in {"정상", "주의", "정지"}:
        raise HTTPException(status_code=400, detail="허용되지 않은 상태입니다.")

    if payload.status == "정지" and target_user.id == admin.user.id:
        raise HTTPException(status_code=400, detail="본인 계정은 정지할 수 없습니다.")

    if payload.status == "정지":
        target_role = await db.scalar(select(AdminRole).where(AdminRole.user_id == target_user.id))
        if target_role and target_role.can_manage_admins:
            if await _count_root_admins(db) <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="마지막 ROOT 관리자는 정지할 수 없습니다.",
                )
        target_user.is_active = False
        target_user.banned_until = datetime.now(timezone.utc) + timedelta(days=30)
    else:
        target_user.is_active = True
        target_user.banned_until = None

    db.add(
        ModerationAction(
            user_id=target_user.id,
            admin_id=admin.user.id,
            action_type=f"STATUS_{payload.status}",
            reason=payload.reason,
        )
    )
    db.add(
        Notification(
            user_id=target_user.id,
            type="SYSTEM",
            title="계정 상태 변경",
            message=f"관리자에 의해 계정 상태가 '{payload.status}'로 변경되었습니다.",
            created_by=admin.user.id,
        )
    )
    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="user_status_updated",
        description=f"{target_user.nickname} 상태를 {payload.status}로 변경",
        path=f"/api/admin/users/{user_id}/status",
    )
    await _append_system_log(
        db,
        level="INFO",
        service="admin",
        message=f"사용자 상태 변경: {target_user.nickname} -> {payload.status}",
        actor=admin.user.nickname,
    )
    await db.commit()
    await db.refresh(target_user)

    report_count = await db.scalar(
        select(func.count())
        .select_from(Report)
        .where(func.lower(Report.target_type) == "user", Report.target_id == target_user.id)
    ) or 0
    party_count = await db.scalar(
        select(func.count()).select_from(PartyMember).where(PartyMember.user_id == target_user.id)
    ) or 0

    return AdminUserRecordOut(
        id=str(target_user.id),
        nickname=target_user.nickname,
        status=_user_status_label(target_user, int(report_count), payload.status),
        reportCount=int(report_count),
        partyCount=int(party_count),
        trustScore=_to_int(target_user.trust_score),
        lastActive=_format_relative(target_user.last_login_at or target_user.updated_at),
    )


@router.get("/parties", response_model=list[AdminPartyRecordOut])
async def get_admin_parties(
    _: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    status_filter: str = Query(default="", alias="status"),
):
    report_counts = _report_target_counts_subquery("party", "party_id")

    stmt = (
        select(Party, Service, User, func.coalesce(report_counts.c.count, 0))
        .join(Service, Party.service_id == Service.id)
        .join(User, Party.leader_id == User.id)
        .outerjoin(report_counts, report_counts.c.party_id == Party.id)
        .order_by(Party.created_at.desc())
    )
    rows = (await db.execute(stmt)).all()

    q = keyword.lower().strip()
    items: list[AdminPartyRecordOut] = []
    for party, service, user, report_count in rows:
        status_label = _party_status_label(party, int(report_count))
        if status_filter and status_label != status_filter:
            continue
        if q and not (
            q in str(party.id).lower()
            or q in service.name.lower()
            or q in user.nickname.lower()
            or q in status_label.lower()
        ):
            continue

        if status_label == "위험":
            payment_note = "검토 필요"
        elif status_label == "종료 예정":
            payment_note = "종료됨"
        elif status_label == "모집중":
            payment_note = "정산 대기"
        else:
            payment_note = "정상 납부"

        items.append(
            AdminPartyRecordOut(
                id=str(party.id),
                service=service.name,
                leaderId=user.nickname,
                memberCount=party.current_members,
                status=status_label,
                reportCount=int(report_count),
                monthlyAmount=party.monthly_per_person * party.current_members,
                lastPayment=payment_note,
            )
        )

    return items


@router.post("/parties/{party_id}/force-end", response_model=AdminPartyRecordOut)
async def force_end_admin_party(
    party_id: str,
    payload: AdminPartyActionIn,
    admin: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
):
    party = await db.get(Party, party_id)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    party.status = "ended"
    party.end_date = datetime.now(timezone.utc).date()

    members = (
        await db.execute(select(PartyMember).where(PartyMember.party_id == party.id))
    ).scalars().all()
    for member in members:
        db.add(
            Notification(
                user_id=member.user_id,
                type="PARTY",
                title="파티 종료 안내",
                message=f"관리자에 의해 파티가 종료되었습니다. 사유: {payload.reason or '운영 정책 위반'}",
                reference_type="party",
                reference_id=party.id,
                created_by=admin.user.id,
            )
        )

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="party_force_ended",
        description=f"{party.title} 파티 강제 종료",
        path=f"/api/admin/parties/{party_id}/force-end",
    )
    await _append_system_log(
        db,
        level="WARN",
        service="admin",
        message=f"파티 강제 종료: {party.title}",
        actor=admin.user.nickname,
    )
    await db.commit()

    report_count = await db.scalar(
        select(func.count())
        .select_from(Report)
        .where(func.lower(Report.target_type) == "party", Report.target_id == party.id)
    ) or 0
    service = await db.get(Service, party.service_id)
    host = await db.get(User, party.leader_id)

    return AdminPartyRecordOut(
        id=str(party.id),
        service=service.name if service else "-",
        leaderId=host.nickname if host else str(party.leader_id),
        memberCount=party.current_members,
        status=_party_status_label(party, int(report_count)),
        reportCount=int(report_count),
        monthlyAmount=party.monthly_per_person * party.current_members,
        lastPayment="종료됨",
    )


@router.get("/reports", response_model=list[ReportRecordOut])
async def get_admin_reports(
    _: AdminContext = Depends(require_admin_report_permission),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(select(Report).order_by(Report.created_at.desc()))).scalars().all()
    items: list[ReportRecordOut] = []
    for report in rows:
        items.append(
            ReportRecordOut(
                id=str(report.id),
                type=_report_type_label(report.target_type),
                target=str(report.target_id),
                reason=report.category,
                status=_report_status_label(report.status),
                content=report.description or "",
                createdAt=_format_datetime(report.created_at),
            )
        )
    return items


@router.patch("/reports/{report_id}", response_model=ReportRecordOut)
async def update_admin_report_status(
    report_id: str,
    payload: AdminStatusUpdateIn,
    admin: AdminContext = Depends(require_admin_report_permission),
    db: AsyncSession = Depends(get_db),
):
    report = await db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")

    report.status = _report_status_code(payload.status)
    report.resolved_by = admin.user.id
    report.resolved_at = datetime.now(timezone.utc)
    report.resolution = payload.status

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="report_status_updated",
        description=f"{report.id} 신고 상태를 {payload.status}로 변경",
        path=f"/api/admin/reports/{report_id}",
    )
    await db.commit()

    return ReportRecordOut(
        id=str(report.id),
        type=_report_type_label(report.target_type),
        target=str(report.target_id),
        reason=report.category,
        status=_report_status_label(report.status),
        content=report.description or "",
        createdAt=_format_datetime(report.created_at),
    )


@router.get("/receipts", response_model=list[ReceiptRecordOut])
async def get_admin_receipts(
    _: AdminContext = Depends(require_admin_receipt_permission),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(select(Receipt).order_by(Receipt.created_at.desc()))).scalars().all()
    return [
        ReceiptRecordOut(
            id=str(receipt.id),
            userId=str(receipt.user_id),
            partyId=str(receipt.party_id),
            ocrAmount=receipt.ocr_amount,
            status=_receipt_status_label(receipt.status),
            createdAt=_format_datetime(receipt.created_at),
        )
        for receipt in rows
    ]


@router.patch("/receipts/{receipt_id}", response_model=ReceiptRecordOut)
async def update_admin_receipt_status(
    receipt_id: str,
    payload: AdminStatusUpdateIn,
    admin: AdminContext = Depends(require_admin_receipt_permission),
    db: AsyncSession = Depends(get_db),
):
    receipt = await db.get(Receipt, receipt_id)
    if not receipt:
        raise HTTPException(status_code=404, detail="영수증을 찾을 수 없습니다.")

    receipt.status = _receipt_status_code(payload.status)
    receipt.reviewed_by = admin.user.id
    receipt.reviewed_at = datetime.now(timezone.utc)
    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="receipt_status_updated",
        description=f"{receipt.id} 영수증 상태를 {payload.status}로 변경",
        path=f"/api/admin/receipts/{receipt_id}",
    )
    await db.commit()

    return ReceiptRecordOut(
        id=str(receipt.id),
        userId=str(receipt.user_id),
        partyId=str(receipt.party_id),
        ocrAmount=receipt.ocr_amount,
        status=_receipt_status_label(receipt.status),
        createdAt=_format_datetime(receipt.created_at),
    )


@router.get("/settlements", response_model=list[SettlementRecordOut])
async def get_admin_settlements(
    _: AdminContext = Depends(require_admin_settlement_permission),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(select(Settlement).order_by(Settlement.created_at.desc()))).scalars().all()
    return [
        SettlementRecordOut(
            id=str(stl.id),
            partyId=str(stl.party_id),
            leaderId=str(stl.leader_id),
            totalAmount=stl.total_amount,
            memberCount=stl.member_count,
            billingMonth=stl.billing_month,
            status=_settlement_status_label(stl.status),
            createdAt=_format_datetime(stl.created_at),
        )
        for stl in rows
    ]


@router.patch("/settlements/{settlement_id}", response_model=SettlementRecordOut)
async def update_admin_settlement_status(
    settlement_id: str,
    payload: AdminStatusUpdateIn,
    admin: AdminContext = Depends(require_admin_settlement_permission),
    db: AsyncSession = Depends(get_db),
):
    stl = await db.get(Settlement, settlement_id)
    if not stl:
        raise HTTPException(status_code=404, detail="정산 데이터를 찾을 수 없습니다.")

    next_status = _settlement_status_code(payload.status)
    stl.status = next_status
    if next_status == "APPROVED":
        stl.approved_by = admin.user.id
        stl.approved_at = datetime.now(timezone.utc)
    elif next_status == "REJECTED":
        stl.rejected_by = admin.user.id
        stl.rejected_at = datetime.now(timezone.utc)

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="settlement_status_updated",
        description=f"{stl.id} 정산 상태를 {payload.status}로 변경",
        path=f"/api/admin/settlements/{settlement_id}",
    )
    await db.commit()

    return SettlementRecordOut(
        id=str(stl.id),
        partyId=str(stl.party_id),
        leaderId=str(stl.leader_id),
        totalAmount=stl.total_amount,
        memberCount=stl.member_count,
        billingMonth=stl.billing_month,
        status=_settlement_status_label(stl.status),
        createdAt=_format_datetime(stl.created_at),
    )


@router.get("/logs", response_model=list[SystemLogRecordOut])
async def get_admin_logs(
    _: AdminContext = Depends(require_admin_log_permission),
    db: AsyncSession = Depends(get_db),
):
    logs: list[SystemLogRecordOut] = []

    activity_rows = (
        await db.execute(select(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(100))
    ).scalars().all()
    system_rows = (
        await db.execute(select(SystemLog).order_by(SystemLog.created_at.desc()).limit(100))
    ).scalars().all()
    moderation_rows = (
        await db.execute(select(ModerationAction).order_by(ModerationAction.created_at.desc()).limit(100))
    ).scalars().all()

    logs.extend(
        [
            SystemLogRecordOut(
                id=str(row.id),
                timestamp=_format_datetime(row.created_at),
                type="ADMIN_ACTION",
                message=row.description,
                actor=str(row.actor_user_id) if row.actor_user_id else "system",
            )
            for row in activity_rows
        ]
    )
    logs.extend(
        [
            SystemLogRecordOut(
                id=str(row.id),
                timestamp=_format_datetime(row.created_at),
                type=row.level.upper(),
                message=row.message,
                actor=row.actor or row.service,
            )
            for row in system_rows
        ]
    )
    logs.extend(
        [
            SystemLogRecordOut(
                id=str(row.id),
                timestamp=_format_datetime(row.created_at),
                type="ADMIN_ACTION",
                message=f"{row.action_type}: {row.reason or '-'}",
                actor=str(row.admin_id) if row.admin_id else "system",
            )
            for row in moderation_rows
        ]
    )

    logs.sort(key=lambda item: item.timestamp, reverse=True)
    return logs[:200]
