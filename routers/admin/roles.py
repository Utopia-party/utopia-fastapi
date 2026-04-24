from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from core.config import settings
from core.database import get_db
from core.redis_client import redis_client
from core.minio_assets import build_minio_asset_url
from core.security import require_user
from models.admin import (
    ActivityLog,
    AdminRole,
    ModerationAction,
    Receipt,
    Settlement,
    SystemLog,
)
from models.report import Report

from models.notification import Notification
from models.party import Party, PartyChat, PartyMember, Service
from models.payment import Payment
from models.quick_match.request import QuickMatchRequest
from models.refresh_token import RefreshToken
from models.mypage.trust_score import TrustScore
from models.user import User
from schemas.admin import (
    AdminDashboardOut,
    AdminModerationHistoryOut,
    AdminPartyActionIn,
    AdminPartyMemberKickIn,
    AdminPartyMemberOut,
    AdminPartyMemberRoleIn,
    AdminPartyRecordOut,
    AdminPermissionOut,
    ChatModerationLogOut,
    ChatModerationStatsOut,
    DashboardChartOut,
    DashboardRecentActivityOut,
    AdminRoleRecordOut,
    AdminRoleUpdateIn,
    AdminServiceRecordOut,
    AdminServiceUpdateIn,
    AdminStatusUpdateIn,
    AdminReportStatusUpdateIn,
    AdminUserAccessLogOut,
    AdminUserDetailOut,
    AdminUserRecordOut,
    AdminUserStatusLogOut,
    AdminUserTrustHistoryOut,
    AdminUserTrustScoreUpdateIn,
    AdminUserStatusUpdateIn,
    DashboardSeriesPointOut,
    ReceiptRecordOut,
    ReportRecordOut,
    SettlementRecordOut,
    SystemLogRecordOut,
    UserStatusLogOut,
)
from services.notifications.report_notification_service import (
    notify_report_result_to_reporter,
    notify_report_warning_to_target,
    notify_report_penalty_to_target,
)

from .deps import (
    AdminContext,
    require_admin_context,
    require_admin_user_permission,
    require_admin_party_permission,
    require_admin_report_permission,
    require_admin_receipt_permission,
    require_admin_settlement_permission,
    require_admin_payment_permission,
    require_admin_handocr_permission,
    require_admin_log_permission,
    require_admin_moderation_permission,
    require_admin_role_permission,
    _format_datetime, _format_relative, _to_int,
    _date_range_bounds, _format_change, _bucket_labels,
    _shift_comparison_range, _series_label,
    _user_display_name, _actor_display_name,
    _build_trust_history_detail, _moderation_action_label,
    _admin_permissions_for_role, _manual_status_label,
    _user_status_label, _party_status_label,
    _report_status_label, _report_status_code,
    _report_type_label, _report_target_counts_subquery,
    _receipt_status_label, _receipt_status_code,
    _settlement_status_label, _settlement_status_code,
    _append_activity_log, _append_system_log,
    _admin_permissions_payload, _has_any_admin_permission,
    _serialize_admin_permissions, _serialize_admin_role,
    _serialize_admin_service, _report_target_display_map,
    _assert_admin_permission, _latest_user_status_actions_subquery,
    _count_root_admins, _ensure_admin_role,
)

router = APIRouter(prefix="/admin", tags=["admin"])

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
        admin_id=admin.user.id,
    )
    await db.commit()
    await db.refresh(role_row)

    return _serialize_admin_role(role_row, target_user, admin.user)
