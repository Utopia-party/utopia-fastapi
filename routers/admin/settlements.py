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

@router.get("/settlements", response_model=list[SettlementRecordOut])
async def get_admin_settlements(
    _: AdminContext = Depends(require_admin_settlement_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    status_filter: str = Query(default="", alias="status"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    leader_user = aliased(User)
    stmt = (
        select(Settlement, Party, leader_user)
        .join(Party, Settlement.party_id == Party.id)
        .join(leader_user, Settlement.leader_id == leader_user.id)
        .order_by(Settlement.created_at.desc())
    )
    dt_from, dt_to = _date_range_bounds(date_from, date_to)
    if dt_from:
        stmt = stmt.where(Settlement.created_at >= dt_from)
    if dt_to:
        stmt = stmt.where(Settlement.created_at < dt_to)

    rows = (await db.execute(stmt)).all()
    q = keyword.lower().strip()

    items: list[SettlementRecordOut] = []
    for stl, party, leader in rows:
        status_label = _settlement_status_label(stl.status)
        party_name = party.title
        leader_name = _user_display_name(leader)
        if status_filter and status_label != status_filter:
            continue
        if q and not (
            q in str(stl.id).lower()
            or q in str(stl.party_id).lower()
            or q in str(stl.leader_id).lower()
            or q in party_name.lower()
            or q in leader_name.lower()
            or q in (stl.billing_month or "").lower()
            or q in status_label.lower()
        ):
            continue
        items.append(
            SettlementRecordOut(
                id=str(stl.id),
                partyId=str(stl.party_id),
                partyName=party_name,
                leaderId=str(stl.leader_id),
                leaderName=leader_name,
                totalAmount=stl.total_amount,
                memberCount=stl.member_count,
                billingMonth=stl.billing_month,
                status=status_label,
                createdAt=_format_datetime(stl.created_at),
            )
        )
    return items


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
    if next_status == "approved":
        stl.approved_by = admin.user.id
        stl.approved_at = datetime.now(timezone.utc)

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="settlement_status_updated",
        description=f"{stl.id} 정산 상태를 {payload.status}로 변경",
        path=f"/api/admin/settlements/{settlement_id}",
    )
    await db.commit()

    party = await db.get(Party, stl.party_id)
    leader = await db.get(User, stl.leader_id)

    return SettlementRecordOut(
        id=str(stl.id),
        partyId=str(stl.party_id),
        partyName=party.title if party else str(stl.party_id),
        leaderId=str(stl.leader_id),
        leaderName=_user_display_name(leader),
        totalAmount=stl.total_amount,
        memberCount=stl.member_count,
        billingMonth=stl.billing_month,
        status=_settlement_status_label(stl.status),
        createdAt=_format_datetime(stl.created_at),
    )
