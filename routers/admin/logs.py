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

@router.get("/logs", response_model=list[SystemLogRecordOut])
async def get_admin_logs(
    _: AdminContext = Depends(require_admin_log_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    log_type: str = Query(default="", alias="type"),
    actor_type: str = Query(default="", alias="actor_type"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    logs: list[SystemLogRecordOut] = []

    dt_from, dt_to = _date_range_bounds(date_from, date_to)

    activity_stmt = select(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(500)
    system_stmt = select(SystemLog).order_by(SystemLog.created_at.desc()).limit(200)
    moderation_stmt = select(ModerationAction).order_by(ModerationAction.created_at.desc()).limit(200)

    if dt_from:
        activity_stmt = activity_stmt.where(ActivityLog.created_at >= dt_from)
        system_stmt = system_stmt.where(SystemLog.created_at >= dt_from)
        moderation_stmt = moderation_stmt.where(ModerationAction.created_at >= dt_from)
    if dt_to:
        activity_stmt = activity_stmt.where(ActivityLog.created_at < dt_to)
        system_stmt = system_stmt.where(SystemLog.created_at < dt_to)
        moderation_stmt = moderation_stmt.where(ModerationAction.created_at < dt_to)

    activity_rows = (await db.execute(activity_stmt)).scalars().all()
    system_rows = (await db.execute(system_stmt)).scalars().all()
    moderation_rows = (await db.execute(moderation_stmt)).scalars().all()

    activity_rows = [
        row
        for row in activity_rows
        if not (
            row.action_type == "admin_access"
            and ((row.extra_metadata or {}).get("path") == "/api/admin/logs")
        )
    ]

    actor_ids = {
        row.actor_user_id
        for row in activity_rows
        if row.actor_user_id is not None
    }
    actor_ids.update(row.admin_id for row in system_rows if row.admin_id is not None)
    actor_ids.update(row.admin_id for row in moderation_rows if row.admin_id is not None)

    users_by_id: dict[Any, User] = {}
    if actor_ids:
        actor_users = (
            await db.execute(select(User).where(User.id.in_(actor_ids)))
        ).scalars().all()
        users_by_id = {user.id: user for user in actor_users}
    admin_user_ids = set(
        (
            await db.execute(select(AdminRole.user_id).where(AdminRole.user_id.in_(actor_ids)))
        ).scalars().all()
    ) if actor_ids else set()

    logs.extend(
        [
            SystemLogRecordOut(
                id=str(row.id),
                timestamp=_format_datetime(row.created_at),
                type=(
                    "ADMIN_ACTION"
                    if row.actor_user_id in admin_user_ids or row.action_type == "admin_access"
                    else "USER_ACTION"
                    if row.actor_user_id is not None
                    else "SYSTEM"
                ),
                message=row.description,
                actor=_actor_display_name(
                    users_by_id.get(row.actor_user_id),
                    "system",
                ),
                actorType=(
                    "admin"
                    if row.actor_user_id in admin_user_ids or row.action_type == "admin_access"
                    else "user"
                    if row.actor_user_id is not None
                    else "system"
                ),
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
                actor=_actor_display_name(
                    users_by_id.get(row.admin_id),
                    ((row.extra_metadata or {}).get("actor") if row.extra_metadata else None)
                    or row.service,
                ),
                actorType=(
                    "admin"
                    if row.admin_id in admin_user_ids or row.admin_id is not None
                    else "system"
                ),
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
                actor=_actor_display_name(
                    users_by_id.get(row.admin_id),
                    "system",
                ),
                actorType="admin",
            )
            for row in moderation_rows
        ]
    )

    logs.sort(key=lambda item: item.timestamp, reverse=True)

    q = keyword.lower().strip()
    lt = log_type.upper().strip()
    at = actor_type.lower().strip()
    if q or lt or at:
        filtered: list[SystemLogRecordOut] = []
        for log in logs:
            if lt and log.type.upper() != lt:
                continue
            if at and log.actorType.lower() != at:
                continue
            if q and q not in log.actor.lower():
                continue
            filtered.append(log)
        return filtered[:200]

    return logs[:200]
