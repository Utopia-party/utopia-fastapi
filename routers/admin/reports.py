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

@router.get("/reports", response_model=list[ReportRecordOut])
async def get_admin_reports(
    _: AdminContext = Depends(require_admin_report_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    report_type: str = Query(default="", alias="type"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    stmt = select(Report).order_by(Report.created_at.desc())
    dt_from, dt_to = _date_range_bounds(date_from, date_to)
    if dt_from:
        stmt = stmt.where(Report.created_at >= dt_from)
    if dt_to:
        stmt = stmt.where(Report.created_at < dt_to)

    rows = (await db.execute(stmt)).scalars().all()
    target_display_map = await _report_target_display_map(db, rows)
    q = keyword.lower().strip()
    rt = report_type.lower().strip()

    items: list[ReportRecordOut] = []
    for report in rows:
        type_label = _report_type_label(report.target_type)
        status_label = _report_status_label(report.status)
        if rt and type_label != rt and report.target_type.lower() != rt:
            continue
        if q and not (
            q in str(report.id).lower()
            or q in str(report.target_id).lower()
            or q in (report.category or "").lower()
            or q in (report.description or "").lower()
            or q in status_label.lower()
            or q in type_label.lower()
        ):
            continue
        items.append(
            ReportRecordOut(
                id=str(report.id),
                type=type_label,
                target=target_display_map.get((report.target_type.lower(), report.target_id), str(report.target_id)),
                reason=report.category,
                status=status_label,
                content=report.description or "",
                createdAt=_format_datetime(report.created_at),
            )
        )
    return items


@router.patch("/reports/{report_id}", response_model=ReportRecordOut)
async def update_admin_report_status(
    report_id: str,
    payload: AdminReportStatusUpdateIn,
    admin: AdminContext = Depends(require_admin_report_permission),
    db: AsyncSession = Depends(get_db),
):
    report = await db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="신고를 찾을 수 없습니다.")

    next_status = _report_status_code(payload.status)
    next_action_result_code = (payload.actionResultCode or "NONE").strip().upper()
    next_admin_memo = payload.adminMemo.strip() if payload.adminMemo else None

    if next_status not in {"PENDING", "IN_REVIEW", "APPROVED", "REJECTED"}:
        raise HTTPException(status_code=400, detail="유효하지 않은 신고 상태입니다.")

    if next_action_result_code not in {"NONE", "WARNING", "PENALTY"}:
        raise HTTPException(status_code=400, detail="유효하지 않은 처리 결과 코드입니다.")

    report.status = next_status
    report.action_result_code = next_action_result_code
    report.admin_memo = next_admin_memo
    report.reviewed_by = admin.user.id
    report.reviewed_at = datetime.now(timezone.utc)

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="report_status_updated",
        description=(
            f"{report.id} 신고 상태를 {payload.status}로 변경 "
            f"(결과 코드: {next_action_result_code})"
        ),
        path=f"/api/admin/reports/{report_id}",
    )
    await db.commit()
    await db.refresh(report)

    # 신고자: 처리 결과 알림
    await notify_report_result_to_reporter(
        db=db,
        report=report,
    )

    # 피신고자: 경고 알림
    if report.action_result_code == "WARNING":
        await notify_report_warning_to_target(
            db=db,
            report=report,
        )

    # 피신고자: 제재 알림
    elif report.action_result_code == "PENALTY":
        await notify_report_penalty_to_target(
            db=db,
            report=report,
        )

    return ReportRecordOut(
        id=str(report.id),
        type=_report_type_label(report.target_type),
        target=report.target_snapshot_name or str(report.target_id),
        reason=report.category,
        status=_report_status_label(report.status),
        content=report.description or "",
        createdAt=_format_datetime(report.created_at),
    )
