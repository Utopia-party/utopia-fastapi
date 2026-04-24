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

@router.get("/receipts", response_model=list[ReceiptRecordOut])
async def get_admin_receipts(
    _: AdminContext = Depends(require_admin_receipt_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    status_filter: str = Query(default="", alias="status"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    stmt = select(Receipt).order_by(Receipt.created_at.desc())
    dt_from, dt_to = _date_range_bounds(date_from, date_to)
    if dt_from:
        stmt = stmt.where(Receipt.created_at >= dt_from)
    if dt_to:
        stmt = stmt.where(Receipt.created_at < dt_to)

    rows = (await db.execute(stmt)).scalars().all()
    q = keyword.lower().strip()

    items: list[ReceiptRecordOut] = []
    for receipt in rows:
        status_label = _receipt_status_label(receipt.status)
        if status_filter and status_label != status_filter:
            continue
        if q and not (
            q in str(receipt.id).lower()
            or q in str(receipt.user_id).lower()
            or q in str(receipt.party_id).lower()
            or q in status_label.lower()
        ):
            continue
        items.append(
            ReceiptRecordOut(
                id=str(receipt.id),
                userId=str(receipt.user_id),
                partyId=str(receipt.party_id),
                ocrAmount=receipt.ocr_amount,
                status=status_label,
                createdAt=_format_datetime(receipt.created_at),
            )
        )
    return items


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
