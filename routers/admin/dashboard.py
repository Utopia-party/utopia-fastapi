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
    require_admin_dashboard_permission,
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

@router.get("/dashboard", response_model=AdminDashboardOut)
async def get_admin_dashboard(
    _: AdminContext = Depends(require_admin_dashboard_permission),
    db: AsyncSession = Depends(get_db),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    compare_mode: str = Query(default="previous_period"),
):
    if compare_mode not in {"previous_period", "year_over_year"}:
        raise HTTPException(status_code=400, detail="유효하지 않은 비교 기준입니다.")

    today = datetime.now(timezone.utc).date()
    end_day = date_to or today
    start_day = date_from or (end_day - timedelta(days=29))
    if start_day > end_day:
        raise HTTPException(status_code=400, detail="시작일은 종료일보다 늦을 수 없습니다.")

    current_from, current_to = _date_range_bounds(start_day, end_day)
    comparison_start_day, comparison_end_day = _shift_comparison_range(
        start_day,
        end_day,
        compare_mode,
    )
    comparison_from, comparison_to = _date_range_bounds(
        comparison_start_day,
        comparison_end_day,
    )

    current_users = (
        await db.execute(
            select(User).where(User.created_at >= current_from, User.created_at < current_to)
        )
    ).scalars().all()
    comparison_users = (
        await db.execute(
            select(User).where(User.created_at >= comparison_from, User.created_at < comparison_to)
        )
    ).scalars().all()
    payment_event_at = func.coalesce(Payment.paid_at, Payment.created_at)

    current_payments = (
        await db.execute(
            select(Payment).where(payment_event_at >= current_from, payment_event_at < current_to)
        )
    ).scalars().all()
    comparison_payments = (
        await db.execute(
            select(Payment).where(
                payment_event_at >= comparison_from,
                payment_event_at < comparison_to,
            )
        )
    ).scalars().all()
    current_reports = (
        await db.execute(
            select(Report).where(Report.created_at >= current_from, Report.created_at < current_to)
        )
    ).scalars().all()
    comparison_reports = (
        await db.execute(
            select(Report).where(Report.created_at >= comparison_from, Report.created_at < comparison_to)
        )
    ).scalars().all()
    current_settlements = (
        await db.execute(
            select(Settlement).where(
                Settlement.created_at >= current_from,
                Settlement.created_at < current_to,
            )
        )
    ).scalars().all()
    comparison_settlements = (
        await db.execute(
            select(Settlement).where(
                Settlement.created_at >= comparison_from,
                Settlement.created_at < comparison_to,
            )
        )
    ).scalars().all()
    current_parties = (
        await db.execute(
            select(Party).where(Party.created_at >= current_from, Party.created_at < current_to)
        )
    ).scalars().all()
    comparison_parties = (
        await db.execute(
            select(Party).where(
                Party.created_at >= comparison_from,
                Party.created_at < comparison_to,
            )
        )
    ).scalars().all()
    current_quick_match_requests = (
        await db.execute(
            select(QuickMatchRequest).where(
                QuickMatchRequest.created_at >= current_from,
                QuickMatchRequest.created_at < current_to,
            )
        )
    ).scalars().all()
    comparison_quick_match_requests = (
        await db.execute(
            select(QuickMatchRequest).where(
                QuickMatchRequest.created_at >= comparison_from,
                QuickMatchRequest.created_at < comparison_to,
            )
        )
    ).scalars().all()

    current_signups = len(current_users)
    comparison_signups = len(comparison_users)
    current_active_users = sum(1 for user in current_users if user.is_active)
    current_suspended_users = sum(1 for user in current_users if not user.is_active)
    current_admin_users = sum(
        1 for user in current_users if (user.role or "").lower() == "admin"
    )
    current_sales = sum(
        payment.amount for payment in current_payments if (payment.status or "").lower() == "approved"
    )
    comparison_sales = sum(
        payment.amount
        for payment in comparison_payments
        if (payment.status or "").lower() == "approved"
    )
    current_commission = sum(
        payment.commission_amount for payment in current_payments
        if (payment.status or "").lower() == "approved"
    )
    comparison_commission = sum(
        payment.commission_amount for payment in comparison_payments
        if (payment.status or "").lower() == "approved"
    )
    current_reports_count = len(current_reports)
    comparison_reports_count = len(comparison_reports)
    current_pending_settlements = sum(
        1 for settlement in current_settlements if (settlement.status or "").lower() == "pending"
    )
    comparison_pending_settlements = sum(
        1
        for settlement in comparison_settlements
        if (settlement.status or "").lower() == "pending"
    )
    current_parties_count = len(current_parties)
    comparison_parties_count = len(comparison_parties)
    current_quick_match_count = len(current_quick_match_requests)
    comparison_quick_match_count = len(comparison_quick_match_requests)
    suspended_users_count = int(current_suspended_users)

    approved_amount = current_sales
    pending_amount = sum(
        payment.amount for payment in current_payments if (payment.status or "").lower() == "pending"
    )
    rejected_amount = sum(
        payment.amount for payment in current_payments if (payment.status or "").lower() == "rejected"
    )

    bucket_starts, bucket_mode = _bucket_labels(start_day, end_day)
    bucket_index = {bucket: idx for idx, bucket in enumerate(bucket_starts)}

    chart_seed = [
        ("sales", "승인 매출", "승인된 결제 금액 기준 비교 그래프", "currency"),
        ("members", "신규 가입", "선택 기간 신규 가입 수 비교 그래프", "count"),
        ("reports", "신고 접수", "선택 기간 신고 접수 건수 비교 그래프", "count"),
        ("settlements", "정산 대기", "선택 기간 생성된 대기 정산 비교 그래프", "count"),
        ("parties", "파티 생성", "선택 기간 파티 생성 추이 비교 그래프", "count"),
        ("quick_match", "빠른 매칭", "선택 기간 빠른 매칭 요청 추이 비교 그래프", "count"),
    ]
    chart_buckets: dict[str, list[DashboardSeriesPointOut]] = {
        chart_id: [
            DashboardSeriesPointOut(
                label=_series_label(bucket, bucket_mode),
                current=0,
                comparison=0,
            )
            for bucket in bucket_starts
        ]
        for chart_id, _, _, _ in chart_seed
    }

    def _align_bucket(
        value_date: date,
        source_start: date,
    ) -> date:
        if bucket_mode == "day":
            return start_day + timedelta(days=(value_date - source_start).days)

        month_offset = (value_date.year - source_start.year) * 12 + (
            value_date.month - source_start.month
        )
        base_month = start_day.month + month_offset
        year = start_day.year + ((base_month - 1) // 12)
        month = ((base_month - 1) % 12) + 1
        return date(year, month, 1)

    def _current_bucket(value_date: date) -> date:
        if bucket_mode == "day":
            return start_day + timedelta(days=(value_date - start_day).days)
        return date(value_date.year, value_date.month, 1)

    for user in current_users:
        bucket = _current_bucket(user.created_at.astimezone(timezone.utc).date())
        if bucket in bucket_index:
            chart_buckets["members"][bucket_index[bucket]].current += 1
    for user in comparison_users:
        bucket = _align_bucket(
            user.created_at.astimezone(timezone.utc).date(),
            comparison_start_day,
        )
        if bucket in bucket_index:
            chart_buckets["members"][bucket_index[bucket]].comparison += 1

    for report in current_reports:
        bucket = _current_bucket(report.created_at.astimezone(timezone.utc).date())
        if bucket in bucket_index:
            chart_buckets["reports"][bucket_index[bucket]].current += 1
    for report in comparison_reports:
        bucket = _align_bucket(
            report.created_at.astimezone(timezone.utc).date(),
            comparison_start_day,
        )
        if bucket in bucket_index:
            chart_buckets["reports"][bucket_index[bucket]].comparison += 1

    for settlement in current_settlements:
        if (settlement.status or "").lower() != "pending":
            continue
        bucket = _current_bucket(settlement.created_at.astimezone(timezone.utc).date())
        if bucket in bucket_index:
            chart_buckets["settlements"][bucket_index[bucket]].current += 1
    for settlement in comparison_settlements:
        if (settlement.status or "").lower() != "pending":
            continue
        bucket = _align_bucket(
            settlement.created_at.astimezone(timezone.utc).date(),
            comparison_start_day,
        )
        if bucket in bucket_index:
            chart_buckets["settlements"][bucket_index[bucket]].comparison += 1

    for party in current_parties:
        bucket = _current_bucket(party.created_at.astimezone(timezone.utc).date())
        if bucket in bucket_index:
            chart_buckets["parties"][bucket_index[bucket]].current += 1
    for party in comparison_parties:
        bucket = _align_bucket(
            party.created_at.astimezone(timezone.utc).date(),
            comparison_start_day,
        )
        if bucket in bucket_index:
            chart_buckets["parties"][bucket_index[bucket]].comparison += 1

    for request in current_quick_match_requests:
        bucket = _current_bucket(request.created_at.astimezone(timezone.utc).date())
        if bucket in bucket_index:
            chart_buckets["quick_match"][bucket_index[bucket]].current += 1
    for request in comparison_quick_match_requests:
        bucket = _align_bucket(
            request.created_at.astimezone(timezone.utc).date(),
            comparison_start_day,
        )
        if bucket in bucket_index:
            chart_buckets["quick_match"][bucket_index[bucket]].comparison += 1

    for payment in current_payments:
        if (payment.status or "").lower() != "approved":
            continue
        payment_at = (payment.paid_at or payment.created_at).astimezone(timezone.utc).date()
        bucket = _current_bucket(payment_at)
        if bucket in bucket_index:
            chart_buckets["sales"][bucket_index[bucket]].current += payment.amount

    for payment in comparison_payments:
        if (payment.status or "").lower() != "approved":
            continue
        payment_at = (payment.paid_at or payment.created_at).astimezone(timezone.utc).date()
        bucket = _align_bucket(
            payment_at,
            comparison_start_day,
        )
        if bucket in bucket_index:
            chart_buckets["sales"][bucket_index[bucket]].comparison += payment.amount

    chart_groups = [
        DashboardChartOut(
            id=chart_id,
            label=label,
            description=description,
            unit=unit,
            points=chart_buckets[chart_id],
        )
        for chart_id, label, description, unit in chart_seed
    ]

    recent_activity_rows = (
        await db.execute(
            select(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(5)
        )
    ).scalars().all()
    activity_actor_ids = {row.actor_user_id for row in recent_activity_rows if row.actor_user_id is not None}
    activity_users: dict[Any, User] = {}
    if activity_actor_ids:
        user_rows = (
            await db.execute(select(User).where(User.id.in_(activity_actor_ids)))
        ).scalars().all()
        activity_users = {user.id: user for user in user_rows}

    signup_delta, signup_trend = _format_change(current_signups, comparison_signups)
    sales_delta, sales_trend = _format_change(current_sales, comparison_sales)
    commission_delta, commission_trend = _format_change(current_commission, comparison_commission)
    report_delta, report_trend = _format_change(current_reports_count, comparison_reports_count)
    settlement_delta, settlement_trend = _format_change(
        current_pending_settlements,
        comparison_pending_settlements,
    )
    party_delta, party_trend = _format_change(current_parties_count, comparison_parties_count)
    quick_match_delta, quick_match_trend = _format_change(
        current_quick_match_count,
        comparison_quick_match_count,
    )
    comparison_label = (
        "전년 동기 비교"
        if compare_mode == "year_over_year"
        else "직전 동일 기간 비교"
    )

    return AdminDashboardOut(
        metrics=[
            {
                "id": "members",
                "label": "신규 가입",
                "value": f"{current_signups:,}",
                "helper": "선택 기간 내 신규 가입 수",
                "delta": signup_delta,
                "trend": signup_trend,
            },
            {
                "id": "sales",
                "label": "승인 매출",
                "value": f"₩ {int(current_sales):,}",
                "helper": "승인된 결제 기준 매출 합계",
                "delta": sales_delta,
                "trend": sales_trend,
            },
            {
                "id": "commission",
                "label": "수수료 수익",
                "value": f"₩ {int(current_commission):,}",
                "helper": "승인된 결제 기준 수수료 합계 (10%)",
                "delta": commission_delta,
                "trend": commission_trend,
            },
            {
                "id": "reports",
                "label": "신고 접수",
                "value": f"{current_reports_count:,}",
                "helper": "선택 기간 내 신규 신고 건수",
                "delta": report_delta,
                "trend": report_trend,
            },
            {
                "id": "settlements",
                "label": "정산 대기",
                "value": f"{current_pending_settlements:,}",
                "helper": "선택 기간 내 생성된 대기 정산",
                "delta": settlement_delta,
                "trend": settlement_trend,
            },
            {
                "id": "parties",
                "label": "파티 생성",
                "value": f"{current_parties_count:,}",
                "helper": "선택 기간 내 새로 만들어진 파티 수",
                "delta": party_delta,
                "trend": party_trend,
            },
            {
                "id": "quick_match",
                "label": "빠른 매칭",
                "value": f"{current_quick_match_count:,}",
                "helper": "선택 기간 내 빠른 매칭 요청 수",
                "delta": quick_match_delta,
                "trend": quick_match_trend,
            },
        ],
        member_stats=[
            {"label": "선택 기간 가입", "value": f"{current_signups:,}"},
            {"label": "활성 사용자", "value": f"{current_active_users:,}"},
            {"label": "정지 사용자", "value": f"{current_suspended_users:,}"},
            {"label": "관리자 계정", "value": f"{current_admin_users:,}"},
        ],
        sales_stats=[
            {"label": "승인 금액", "value": f"₩ {int(approved_amount):,}"},
            {"label": "대기 금액", "value": f"₩ {int(pending_amount):,}"},
            {"label": "거절 금액", "value": f"₩ {int(rejected_amount):,}"},
            {"label": "파티 생성", "value": f"{current_parties_count:,}건"},
            {"label": "빠른 매칭", "value": f"{current_quick_match_count:,}건"},
            {"label": "정지/제재", "value": f"{suspended_users_count:,}건"},
            {
                "label": "비교 기준",
                "value": (
                    f"{comparison_start_day.isoformat()} ~ {comparison_end_day.isoformat()}"
                ),
            },
        ],
        today_summary=(
            f"{start_day.isoformat()} ~ {end_day.isoformat()} 기준 "
            f"가입 {current_signups}건 / 신고 {current_reports_count}건 / 승인 매출 ₩ {int(current_sales):,}"
        ),
        period_label=f"{start_day.isoformat()} ~ {end_day.isoformat()}",
        comparison_label=comparison_label,
        compare_mode=compare_mode,
        range_start=start_day.isoformat(),
        range_end=end_day.isoformat(),
        chart_points=chart_buckets["sales"],
        chart_groups=chart_groups,
        recent_activities=[
            DashboardRecentActivityOut(
                timestamp=_format_datetime(row.created_at),
                title=row.action_type,
                description=f"{_actor_display_name(activity_users.get(row.actor_user_id), 'system')} · {row.description}",
            )
            for row in recent_activity_rows
        ],
    )
