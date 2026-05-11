from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel as _BaseModel
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




def _extract_discount_rate(discount_reason: str | None, service: Service | None) -> float:
    if not discount_reason:
        return 0.0

    total = 0.0

    if "방장 할인" in discount_reason and service and service.leader_discount_rate is not None:
        total += float(service.leader_discount_rate)

    if "추천인 할인" in discount_reason and service and service.referral_discount_rate is not None:
        total += float(service.referral_discount_rate)

    if total > 0:
        return min(total, 1.0)

    matches = re.findall(r"(\d+)%", discount_reason)
    if not matches:
        return 0.0
    return min(sum(int(match) for match in matches) / 100, 1.0)

class AdminPaymentRecordOut(_BaseModel):
    id: str
    userId: str
    userNickname: str
    userName: str | None
    partyId: str
    partyTitle: str
    serviceName: str | None
    role: str
    basePrice: int
    amount: int
    discountReason: str | None
    statusReason: str | None
    baseCommissionRate: float
    commissionRate: float
    effectiveCommissionRate: float
    commissionAmount: int
    appliedLeaderDiscountRate: float
    appliedReferralDiscountRate: float
    paymentMethod: str | None
    status: str
    billingMonth: str
    pricingType: str | None
    quickMatchFeeRate: float
    paidAt: str | None
    createdAt: str

    class Config:
        from_attributes = True


class AdminPaymentListOut(_BaseModel):
    items: list[AdminPaymentRecordOut]
    total: int
    page: int
    limit: int
    totalPages: int


class AdminPaymentStatusUpdateIn(_BaseModel):
    status: str
    reason: str | None = None


async def _get_party_active_member_ids(
    db: AsyncSession,
    party: Party,
) -> list[Any]:
    member_ids = (
        await db.execute(
            select(PartyMember.user_id).where(
                PartyMember.party_id == party.id,
                PartyMember.status == "active",
            )
        )
    ).scalars().all()
    if party.leader_id and party.leader_id not in member_ids:
        member_ids.append(party.leader_id)
    return member_ids


async def _sync_related_settlement_status(
    db: AsyncSession,
    party: Party,
    billing_month: str,
) -> tuple[Settlement | None, bool]:
    settlement = (
        await db.execute(
            select(Settlement).where(
                Settlement.party_id == party.id,
                Settlement.billing_month == billing_month,
            )
        )
    ).scalar_one_or_none()
    if settlement is None:
        return None, False

    member_ids = await _get_party_active_member_ids(db, party)
    paid_ids = set(
        (
            await db.execute(
                select(Payment.user_id).where(
                    Payment.party_id == party.id,
                    Payment.billing_month == billing_month,
                    Payment.status == "approved",
                )
            )
        ).scalars().all()
    )
    all_paid = all(member_id in paid_ids for member_id in member_ids)

    auto_approved = False
    if all_paid and settlement.status != "approved":
        settlement.status = "approved"
        settlement.approved_at = datetime.now(timezone.utc)
        auto_approved = True
    elif not all_paid and settlement.status in {"approved", "pending"}:
        await db.delete(settlement)
        return None, False

    return settlement, auto_approved


def _payment_status_label(value: str) -> str:
    return {
        "pending": "대기",
        "approved": "승인",
        "rejected": "거절",
        "completed": "완료",
        "failed": "실패",
        "cancelled": "취소",
        "refunded": "환불",
    }.get(value.lower(), value)


def _payment_status_code(value: str) -> str:
    return {
        "대기": "pending",
        "승인": "approved",
        "거절": "rejected",
        "완료": "completed",
        "실패": "failed",
        "취소": "cancelled",
        "환불": "refunded",
    }.get(value, value.lower())


def _build_admin_payment_record(
    payment: Payment,
    user: User,
    party: Party,
    service: Service | None,
) -> AdminPaymentRecordOut:
    role = "방장" if party.leader_id == user.id else "멤버"

    base_price = int(payment.base_price) if payment.base_price else int(payment.amount)
    discount_rate = _extract_discount_rate(payment.discount_reason, service)
    expected_amount = round(base_price * (1 - discount_rate))
    amount = expected_amount if discount_rate > 0 else int(payment.amount)

    commission_rate = float(payment.commission_rate or 0)
    commission_amount = int(payment.commission_amount or 0)
    applied_leader_discount_rate = (
        float(service.leader_discount_rate or 0)
        if payment.discount_reason and "방장 할인" in payment.discount_reason and service
        else 0.0
    )
    applied_referral_discount_rate = (
        float(service.referral_discount_rate or 0)
        if payment.discount_reason and "추천인 할인" in payment.discount_reason and service
        else 0.0
    )

    quick_match_fee_rate = 0.0
    if payment.pricing_type == "quick_match":
        if service and service.quick_match_fee_rate is not None:
            quick_match_fee_rate = float(service.quick_match_fee_rate or 0)
        elif service and service.commission_rate is not None:
            quick_match_fee_rate = max(
                commission_rate - float(service.commission_rate or 0),
                0.0,
            )

    base_commission_rate = max(commission_rate - quick_match_fee_rate, 0.0)
    effective_commission_rate = max(
        commission_rate
        - applied_leader_discount_rate
        - applied_referral_discount_rate,
        0.0,
    )

    return AdminPaymentRecordOut(
        id=str(payment.id),
        userId=str(user.id),
        userNickname=user.nickname,
        userName=user.name,
        partyId=str(party.id),
        partyTitle=party.title,
        serviceName=service.name if service else None,
        role=role,
        basePrice=base_price,
        amount=amount,
        discountReason=payment.discount_reason,
        statusReason=payment.status_reason,
        baseCommissionRate=base_commission_rate,
        commissionRate=commission_rate,
        effectiveCommissionRate=effective_commission_rate,
        commissionAmount=commission_amount,
        appliedLeaderDiscountRate=applied_leader_discount_rate,
        appliedReferralDiscountRate=applied_referral_discount_rate,
        paymentMethod=payment.payment_method,
        status=payment.status,
        billingMonth=payment.billing_month,
        pricingType=payment.pricing_type,
        quickMatchFeeRate=quick_match_fee_rate,
        paidAt=payment.paid_at.isoformat() if payment.paid_at else None,
        createdAt=payment.created_at.isoformat(),
    )


@router.get("/payments", response_model=AdminPaymentListOut)
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
        items.append(_build_admin_payment_record(payment, user, party, service))

    return AdminPaymentListOut(
        items=items,
        total=total,
        page=page,
        limit=limit,
        totalPages=total_pages,
    )


@router.patch("/payments/{payment_id}", response_model=AdminPaymentRecordOut)
async def update_admin_payment_status(
    payment_id: str,
    payload: AdminPaymentStatusUpdateIn,
    admin: AdminContext = Depends(require_admin_payment_permission),
    db: AsyncSession = Depends(get_db),
):
    payment = await db.get(Payment, payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="결제 데이터를 찾을 수 없습니다.")

    next_status = _payment_status_code(payload.status)
    if next_status not in {"pending", "approved", "rejected", "cancelled"}:
        raise HTTPException(status_code=400, detail="결제 상태는 대기, 승인, 거절, 취소만 변경할 수 있습니다.")

    reason = (payload.reason or "").strip() or None
    if next_status in {"rejected", "cancelled"} and not reason:
        raise HTTPException(status_code=400, detail="사유를 입력해주세요.")

    payment.status = next_status
    if next_status == "approved":
        payment.paid_at = payment.paid_at or datetime.now(timezone.utc)
        payment.status_reason = None
    else:
        payment.paid_at = None
        payment.status_reason = reason

    party = await db.get(Party, payment.party_id)
    settlement, auto_approved = (None, False)
    if party:
        settlement, auto_approved = await _sync_related_settlement_status(
            db,
            party,
            payment.billing_month,
        )

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="payment_status_updated",
        description=f"{payment.id} 결제 상태를 {_payment_status_label(next_status)}로 변경",
        path=f"/api/admin/payments/{payment_id}",
    )
    await db.commit()

    if party:
        try:
            from routers.parties import _broadcast_party_updated

            await _broadcast_party_updated(party.id)
        except Exception:
            pass

    if auto_approved and settlement and party:
        from services.notification_service import notify_user

        await notify_user(
            db=db,
            user_id=settlement.leader_id,
            type="settlement",
            title="정산 자동 승인 완료",
            message=f"[{party.title}] 모든 멤버 결제가 확인되어 정산이 자동 승인되었습니다. 아이디/비밀번호를 공유해주세요.",
            reference_type="settlement",
            reference_id=party.id,
            metadata={
                "event_code": "SETTLEMENT_AUTO_APPROVED_FROM_PAYMENT",
                "party_id": str(party.id),
                "settlement_id": str(settlement.id),
            },
        )
        try:
            from routers.chat import manager

            await manager.broadcast(
                str(party.id),
                {
                    "type": "settlement_approved",
                    "party_id": str(party.id),
                    "settlement_id": str(settlement.id),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            pass

    if next_status in {"rejected", "cancelled"} and party:
        from services.notification_service import notify_user

        title = (
            "결제가 취소되었습니다"
            if next_status == "cancelled"
            else "결제가 거절되었습니다"
        )
        action_label = "취소" if next_status == "cancelled" else "거절"
        await notify_user(
            db=db,
            user_id=payment.user_id,
            type="payment",
            title=title,
            message=f"[{party.title}] 이번 달 결제가 {action_label}되었습니다. 사유: {reason}. 다시 결제를 진행해주세요.",
            reference_type="payment",
            reference_id=payment.id,
            metadata={
                "event_code": (
                    "PAYMENT_CANCELLED_BY_ADMIN"
                    if next_status == "cancelled"
                    else "PAYMENT_REJECTED_BY_ADMIN"
                ),
                "party_id": str(party.id),
                "payment_id": str(payment.id),
                "reason": reason,
            },
        )

    stmt = (
        select(Payment, User, Party, Service)
        .join(User, Payment.user_id == User.id)
        .join(Party, Payment.party_id == Party.id)
        .outerjoin(Service, Party.service_id == Service.id)
        .where(Payment.id == payment.id)
    )
    row = (await db.execute(stmt)).first()
    if not row:
        raise HTTPException(status_code=404, detail="결제 데이터를 다시 불러오지 못했습니다.")

    current_payment, user, party, service = row
    return _build_admin_payment_record(current_payment, user, party, service)
