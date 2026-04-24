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

async def get_admin_payments(
    keyword: str = Query(""),
    status_filter: str = Query("", alias="status"),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    _: AdminContext = Depends(require_admin_payment_permission),
    db: AsyncSession = Depends(get_db),
):
    """결제 내역 전체 목록 (관리자 전용, 페이지네이션)"""
    payment_event_at = func.coalesce(Payment.paid_at, Payment.created_at)

    stmt = (
        select(Payment, User, Party, Service)
        .join(User, Payment.user_id == User.id)
        .join(Party, Payment.party_id == Party.id)
        .outerjoin(Service, Party.service_id == Service.id)
        .order_by(payment_event_at.desc())
    )

    if status_filter:
        stmt = stmt.where(func.lower(Payment.status) == status_filter.lower())
    if date_from:
        stmt = stmt.where(func.date(payment_event_at) >= date_from)
    if date_to:
        stmt = stmt.where(func.date(payment_event_at) <= date_to)

    rows = (await db.execute(stmt)).all()

    # 키워드 필터 (Python-side)
    filtered = []
    for payment, user, party, service in rows:
        if keyword:
            kw = keyword.lower()
            hit = (
                kw in (user.nickname or "").lower()
                or kw in (user.name or "").lower()
                or kw in (party.title or "").lower()
                or kw in (service.name if service else "").lower()
            )
            if not hit:
                continue
        filtered.append((payment, user, party, service))

    total = len(filtered)
    total_pages = max(1, (total + limit - 1) // limit)
    paginated = filtered[(page - 1) * limit : page * limit]

    items = []
    for payment, user, party, service in paginated:
        role = "방장" if party.leader_id == user.id else "멤버"
        base_price, actual_amount = _admin_payment_display_amount(
            payment, user, party, service
        )
        items.append(
            AdminPaymentRecordOut(
                id=str(payment.id),
                userId=str(user.id),
                userNickname=user.nickname,
                userName=user.name,
                partyId=str(party.id),
                partyTitle=party.title,
                serviceName=service.name if service else None,
                role=role,
                basePrice=base_price,
                amount=actual_amount,
                discountReason=payment.discount_reason,
                commissionRate=float(payment.commission_rate or 0.10),
                commissionAmount=round(
                    actual_amount * float(payment.commission_rate or 0.10)
                ),
                paymentMethod=payment.payment_method,
                status=payment.status,
                billingMonth=payment.billing_month,
                pricingType=payment.pricing_type,
                paidAt=payment.paid_at.isoformat() if payment.paid_at else None,
                createdAt=payment.created_at.isoformat(),
            )
        )

    return AdminPaymentListOut(
        items=items,
        total=total,
        page=page,
        limit=limit,
        totalPages=total_pages,
    )

# ── 캡챠 통계 (대시보드) ──────────────────────────────────────

@router.get("/captcha/stats", tags=["admin-captcha"])