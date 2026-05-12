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
    SettlementParticipantPaymentOut,
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


def _settlement_payment_status_label(value: str | None) -> str:
    return {
        "approved": "승인",
        "pending": "대기",
        "rejected": "거절",
        "cancelled": "취소",
    }.get((value or "").lower(), "대기")


async def _build_settlement_participant_payments_bulk(
    db: AsyncSession,
    party_ids: list,
    billing_months_by_party: dict,
) -> dict[str, list[SettlementParticipantPaymentOut]]:
    """
    N+1 해소: 전체 party_id 목록을 한 번에 IN 쿼리로 조회.
    반환값: { party_id(str): [SettlementParticipantPaymentOut, ...] }
    """
    if not party_ids:
        return {}

    # ── 1) 파티 멤버 + 유저 일괄 조회 ─────────────────────────────
    member_rows = (
        await db.execute(
            select(PartyMember, User)
            .join(User, PartyMember.user_id == User.id)
            .where(
                PartyMember.party_id.in_(party_ids),
                PartyMember.status == "active",
            )
        )
    ).all()

    # party_id → [(member, user), ...]
    members_by_party: dict = {}
    for member, user in member_rows:
        members_by_party.setdefault(str(member.party_id), []).append((member, user))

    # ── 2) 결제 상태 일괄 조회 ────────────────────────────────────
    # billing_month 는 파티마다 다를 수 있으므로 (party_id, billing_month) OR 조건 사용
    from sqlalchemy import tuple_
    party_month_pairs = [
        (pid, billing_months_by_party[pid]) for pid in party_ids
        if pid in billing_months_by_party
    ]
    payment_rows = (
        await db.execute(
            select(Payment.party_id, Payment.user_id, Payment.status).where(
                tuple_(Payment.party_id, Payment.billing_month).in_(party_month_pairs)
            )
        )
    ).all() if party_month_pairs else []

    # (party_id, user_id) → status
    payment_status_map: dict = {}
    for p_party_id, p_user_id, p_status in payment_rows:
        payment_status_map[(str(p_party_id), str(p_user_id))] = p_status

    # ── 3) 리더가 멤버 테이블에 없는 케이스 보완 ─────────────────
    # 누락된 leader_id 를 모아 한 번에 조회
    from models.party import Party as PartyModel
    party_leader_rows = (
        await db.execute(
            select(PartyModel.id, PartyModel.leader_id).where(
                PartyModel.id.in_(party_ids)
            )
        )
    ).all()
    leader_id_by_party = {str(row.id): str(row.leader_id) for row in party_leader_rows if row.leader_id}

    missing_leader_ids = []
    for pid in party_ids:
        leader_id = leader_id_by_party.get(pid)
        if not leader_id:
            continue
        members = members_by_party.get(pid, [])
        member_user_ids = {str(u.id) for _, u in members}
        if leader_id not in member_user_ids:
            missing_leader_ids.append(leader_id)

    missing_leader_map: dict[str, Any] = {}
    if missing_leader_ids:
        missing_users = (
            await db.execute(
                select(User).where(User.id.in_(missing_leader_ids))
            )
        ).scalars().all()
        missing_leader_map = {str(u.id): u for u in missing_users}

    # ── 4) 결과 조립 ─────────────────────────────────────────────
    result: dict[str, list[SettlementParticipantPaymentOut]] = {}
    for pid in party_ids:
        leader_id = leader_id_by_party.get(pid)
        members = members_by_party.get(pid, [])
        member_user_ids = {str(u.id) for _, u in members}
        rows_out: list[SettlementParticipantPaymentOut] = []

        for member, user in members:
            rows_out.append(
                SettlementParticipantPaymentOut(
                    userId=str(user.id),
                    nickname=user.nickname,
                    role="파티장" if member.role == "leader" else "멤버",
                    paymentStatus=_settlement_payment_status_label(
                        payment_status_map.get((pid, str(user.id)))
                    ),
                )
            )

        # 리더가 멤버 테이블에 없는 경우 보완
        if leader_id and leader_id not in member_user_ids:
            leader_user = missing_leader_map.get(leader_id)
            if leader_user:
                rows_out.append(
                    SettlementParticipantPaymentOut(
                        userId=str(leader_user.id),
                        nickname=leader_user.nickname,
                        role="파티장",
                        paymentStatus=_settlement_payment_status_label(
                            payment_status_map.get((pid, str(leader_user.id)))
                        ),
                    )
                )

        rows_out.sort(key=lambda item: (0 if item.role == "파티장" else 1, item.nickname))
        result[pid] = rows_out

    return result


@router.get("/settlements", response_model=list[SettlementRecordOut])
async def get_admin_settlements(
    _: AdminContext = Depends(require_admin_settlement_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    status_filter: str = Query(default="", alias="status"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    from sqlalchemy import or_
    leader_user = aliased(User)
    stmt = (
        select(Settlement, Party, leader_user)
        .join(Party, Settlement.party_id == Party.id)
        .join(leader_user, Settlement.leader_id == leader_user.id)
        .order_by(Settlement.created_at.desc())
    )

    # ── 날짜 필터: DB에서 처리 ────────────────────────────────────
    dt_from, dt_to = _date_range_bounds(date_from, date_to)
    if dt_from:
        stmt = stmt.where(Settlement.created_at >= dt_from)
    if dt_to:
        stmt = stmt.where(Settlement.created_at < dt_to)

    # ── status 필터: DB에서 처리 ──────────────────────────────────
    if status_filter:
        status_code = _settlement_status_code(status_filter)
        if status_code:
            stmt = stmt.where(Settlement.status == status_code)

    # ── keyword 필터: DB에서 처리 ─────────────────────────────────
    q = keyword.strip()
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            or_(
                Party.title.ilike(like),
                leader_user.nickname.ilike(like),
                Settlement.billing_month.ilike(like),
            )
        )

    rows = (await db.execute(stmt)).all()

    # ── N+1 해소: 참여자 결제 상태 일괄 조회 ─────────────────────
    party_ids = [str(stl.party_id) for stl, _, _ in rows]
    billing_months_by_party = {
        str(stl.party_id): stl.billing_month for stl, _, _ in rows
    }
    participant_map = await _build_settlement_participant_payments_bulk(
        db, party_ids, billing_months_by_party
    )

    items: list[SettlementRecordOut] = []
    for stl, party, leader in rows:
        pid = str(stl.party_id)
        items.append(
            SettlementRecordOut(
                id=str(stl.id),
                partyId=pid,
                partyName=party.title,
                leaderId=str(stl.leader_id),
                leaderName=leader.nickname,
                totalAmount=stl.total_amount,
                memberCount=stl.member_count,
                billingMonth=stl.billing_month,
                status=_settlement_status_label(stl.status),
                createdAt=_format_datetime(stl.created_at),
                participantPayments=participant_map.get(pid, []),
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

    # 관리자 approved 처리 → 방장 알림 + WebSocket 브로드캐스트
    if next_status == "approved" and party:
        from services.notification_service import notify_user
        await notify_user(
            db=db,
            user_id=stl.leader_id,
            type="settlement",
            title="정산 승인 완료 (관리자)",
            message=f"[{party.title}] 정산이 관리자에 의해 승인되었습니다. 아이디/비밀번호를 공유해주세요.",
            reference_type="settlement",
            reference_id=party.id,
            metadata={
                "event_code": "SETTLEMENT_ADMIN_APPROVED",
                "party_id": str(party.id),
                "settlement_id": str(stl.id),
            },
        )
        try:
            from routers.chat import manager
            await manager.broadcast(str(party.id), {
                "type": "settlement_approved",
                "party_id": str(party.id),
                "settlement_id": str(stl.id),
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:
            pass

    return SettlementRecordOut(
        id=str(stl.id),
        partyId=str(stl.party_id),
        partyName=party.title if party else str(stl.party_id),
        leaderId=str(stl.leader_id),
        leaderName=leader.nickname if leader else str(stl.leader_id),
        totalAmount=stl.total_amount,
        memberCount=stl.member_count,
        billingMonth=stl.billing_month,
        status=_settlement_status_label(stl.status),
        createdAt=_format_datetime(stl.created_at),
        participantPayments=(
            (
                await _build_settlement_participant_payments_bulk(
                    db,
                    [str(stl.party_id)],
                    {str(stl.party_id): stl.billing_month},
                )
            ).get(str(stl.party_id), [])
            if party
            else []
        ),
    )
