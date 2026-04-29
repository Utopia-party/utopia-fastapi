from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
import uuid

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
from services.notifications.party_notification_service import (
    notify_party_member_kicked,
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


async def _recalculate_party_member_count(db: AsyncSession, party: Party) -> int:
    active_user_ids = set(
        (
            await db.execute(
                select(PartyMember.user_id).where(
                    PartyMember.party_id == party.id,
                    PartyMember.status == "active",
                )
            )
        ).scalars().all()
    )
    if party.leader_id:
        active_user_ids.add(party.leader_id)
    return len(active_user_ids)

@router.get("/parties", response_model=list[AdminPartyRecordOut])
async def get_admin_parties(
    _: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    status_filter: str = Query(default="", alias="status"),
    category_filter: str = Query(default="", alias="category"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    report_counts = _report_target_counts_subquery("party", "party_id")

    stmt = (
        select(Party, Service, User, func.coalesce(report_counts.c.count, 0))
        .join(Service, Party.service_id == Service.id)
        .join(User, Party.leader_id == User.id)
        .outerjoin(report_counts, report_counts.c.party_id == Party.id)
        .order_by(Party.created_at.desc())
    )
    dt_from, dt_to = _date_range_bounds(date_from, date_to)
    if dt_from:
        stmt = stmt.where(Party.created_at >= dt_from)
    if dt_to:
        stmt = stmt.where(Party.created_at < dt_to)
    if category_filter.strip():
        stmt = stmt.where(func.lower(Service.category) == category_filter.strip().lower())

    rows = (await db.execute(stmt)).all()

    q = keyword.lower().strip()
    category_q = category_filter.lower().strip()
    items: list[AdminPartyRecordOut] = []
    for party, service, user, report_count in rows:
        status_label = _party_status_label(party, int(report_count))
        if status_filter and status_label != status_filter:
            continue
        if category_q and category_q != (service.category or "").lower():
            continue
        if q and not (
            q in str(party.id).lower()
            or q in party.title.lower()
            or q in service.name.lower()
            or q in (service.category or "").lower()
            or q in user.nickname.lower()
            or q in status_label.lower()
        ):
            continue

        if status_label == "위험":
            payment_note = "검토 필요"
        # 파티 종료 수정
        elif status_label == "종료됨":
            payment_note = "종료됨"
        # 파티 종료 수정
        elif status_label == "모집중":
            payment_note = "정산 대기"
        else:
            payment_note = "정상 납부"

        items.append(
            AdminPartyRecordOut(
                id=str(party.id),
                title=party.title,
                createdAt=_format_datetime(party.created_at),
                service=service.name,
                category=service.category,
                leaderId=str(user.id),
                leaderNickname=user.nickname,
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
        admin_id=admin.user.id,
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
        title=party.title,
        createdAt=_format_datetime(party.created_at),
        service=service.name if service else "-",
        category=service.category if service else "-",
        leaderId=str(host.id) if host else str(party.leader_id),
        leaderNickname=host.nickname if host else str(party.leader_id),
        memberCount=party.current_members,
        status=_party_status_label(party, int(report_count)),
        reportCount=int(report_count),
        monthlyAmount=party.monthly_per_person * party.current_members,
        lastPayment="종료됨",
    )


@router.get("/parties/{party_id}/members", response_model=list[AdminPartyMemberOut])
async def get_admin_party_members(
    party_id: str,
    _: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
):
    """파티 상세 - 멤버 목록 (활성 + 퇴장 포함)"""
    try:
        party_uuid = __import__("uuid").UUID(party_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="유효하지 않은 파티 ID입니다.")

    party = await db.get(Party, party_uuid)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    rows = (
        await db.execute(
            select(PartyMember, User)
            .join(User, PartyMember.user_id == User.id)
            .where(PartyMember.party_id == party_uuid)
            .order_by(PartyMember.joined_at.asc())
        )
    ).all()

    result: list[AdminPartyMemberOut] = []
    for member, user in rows:
        # 파티장 여부: party.leader_id 또는 role == "leader"
        effective_role = "leader" if user.id == party.leader_id or member.role == "leader" else member.role
        result.append(
            AdminPartyMemberOut(
                memberId=str(member.id),
                userId=str(user.id),
                nickname=user.nickname,
                name=user.name,
                role=effective_role,
                status=member.status,
                trustScore=float(user.trust_score) if user.trust_score is not None else 36.5,
                joinedAt=_format_datetime(member.joined_at),
                leftAt=_format_datetime(member.left_at) if member.left_at else None,
            )
        )
    return result


@router.post("/parties/{party_id}/members/{user_id}/kick", response_model=AdminPartyMemberOut)
async def kick_admin_party_member(
    party_id: str,
    user_id: str,
    payload: AdminPartyMemberKickIn,
    admin: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
):
    """파티 멤버 강퇴 (파티장 강퇴 시 다음 멤버를 파티장으로 승격)"""
    try:
        import uuid as _uuid
        party_uuid = _uuid.UUID(party_id)
        target_uuid = _uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="유효하지 않은 ID입니다.")

    party = await db.get(Party, party_uuid)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    member_row = (
        await db.execute(
            select(PartyMember)
            .where(PartyMember.party_id == party_uuid, PartyMember.user_id == target_uuid, PartyMember.status == "active")
        )
    ).scalar_one_or_none()
    if not member_row:
        raise HTTPException(status_code=404, detail="활성 멤버가 아닙니다.")

    target_user = await db.get(User, target_uuid)

    was_leader = (party.leader_id == target_uuid)

    # 멤버 강퇴 처리
    member_row.status = "kicked"
    member_row.left_at = func.now()

    # 파티장 강퇴 시 → 가장 오래된 active 멤버를 새 파티장으로 승격
    new_leader_user: User | None = None
    if was_leader:
        next_member_row = (
            await db.execute(
                select(PartyMember, User)
                .join(User, PartyMember.user_id == User.id)
                .where(
                    PartyMember.party_id == party_uuid,
                    PartyMember.status == "active",
                    PartyMember.user_id != target_uuid,
                )
                .order_by(PartyMember.joined_at.asc())
                .limit(1)
            )
        ).first()

        if next_member_row:
            next_member, next_user = next_member_row
            next_member.role = "leader"
            party.leader_id = next_user.id
            new_leader_user = next_user
            db.add(
                Notification(
                    user_id=next_user.id,
                    type="PARTY",
                    title="파티장 승계 안내",
                    message=f"관리자 조치로 이전 파티장이 강퇴되어 회원님이 '{party.title}' 파티의 새 파티장이 되었습니다.",
                    reference_type="party",
                    reference_id=party.id,
                )
            )
        else:
            # 멤버가 없으면 파티 종료
            party.status = "ended"
            party.end_date = datetime.now(timezone.utc).date()

    party.current_members = await _recalculate_party_member_count(db, party)

    await notify_party_member_kicked(
        db=db,
        party=party,
        target_user_id=target_uuid,
        reason=payload.reason or "운영 정책 위반",
    )

    desc_extra = f" (신규 파티장: {new_leader_user.nickname})" if new_leader_user else ""
    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="party_member_kicked",
        description=f"{target_user.nickname if target_user else user_id} 파티 강퇴 ({party.title}){desc_extra}",
        path=f"/api/admin/parties/{party_id}/members/{user_id}/kick",
        target_id=target_uuid,
        reason=payload.reason,
    )
    await db.commit()
    await db.refresh(member_row)

    return AdminPartyMemberOut(
        memberId=str(member_row.id),
        userId=str(target_uuid),
        nickname=target_user.nickname if target_user else str(target_uuid),
        name=target_user.name if target_user else None,
        role="leader" if was_leader else member_row.role,
        status=member_row.status,
        trustScore=float(target_user.trust_score) if target_user and target_user.trust_score is not None else 36.5,
        joinedAt=_format_datetime(member_row.joined_at),
        leftAt=_format_datetime(member_row.left_at),
    )


@router.patch("/parties/{party_id}/members/{user_id}/role", response_model=AdminPartyMemberOut)
async def change_admin_party_member_role(
    party_id: str,
    user_id: str,
    payload: AdminPartyMemberRoleIn,
    admin: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
):
    """파티 멤버 역할 변경 (파티장 ↔ 멤버). 파티장 변경 시 기존 파티장은 member로 강등."""
    if payload.role not in {"leader", "member"}:
        raise HTTPException(status_code=400, detail="role은 'leader' 또는 'member'만 허용됩니다.")

    try:
        import uuid as _uuid
        party_uuid = _uuid.UUID(party_id)
        target_uuid = _uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="유효하지 않은 ID입니다.")

    party = await db.get(Party, party_uuid)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    member_row = (
        await db.execute(
            select(PartyMember)
            .where(PartyMember.party_id == party_uuid, PartyMember.user_id == target_uuid, PartyMember.status == "active")
        )
    ).scalar_one_or_none()
    if not member_row:
        raise HTTPException(status_code=404, detail="활성 멤버가 아닙니다.")

    target_user = await db.get(User, target_uuid)

    if payload.role == "leader" and party.leader_id != target_uuid:
        # 기존 파티장 member로 강등
        old_leader_row = (
            await db.execute(
                select(PartyMember)
                .where(PartyMember.party_id == party_uuid, PartyMember.user_id == party.leader_id, PartyMember.status == "active")
            )
        ).scalar_one_or_none()
        if old_leader_row:
            old_leader_row.role = "member"

        old_leader_user = await db.get(User, party.leader_id)
        if old_leader_user:
            db.add(
                Notification(
                    user_id=old_leader_user.id,
                    type="PARTY",
                    title="파티장 변경 안내",
                    message=f"관리자 조치로 '{party.title}' 파티의 파티장이 변경되었습니다.",
                    reference_type="party",
                    reference_id=party.id,
                )
            )

        party.leader_id = target_uuid
        member_row.role = "leader"

        if target_user:
            db.add(
                Notification(
                    user_id=target_uuid,
                    type="PARTY",
                    title="파티장 임명 안내",
                    message=f"관리자에 의해 '{party.title}' 파티의 파티장으로 임명되었습니다.",
                    reference_type="party",
                    reference_id=party.id,
                )
            )
    else:
        member_row.role = payload.role

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="party_member_role_changed",
        description=f"{target_user.nickname if target_user else user_id} 역할 변경 → {payload.role} ({party.title})",
        path=f"/api/admin/parties/{party_id}/members/{user_id}/role",
        target_id=target_uuid,
    )
    await db.commit()
    await db.refresh(member_row)

    return AdminPartyMemberOut(
        memberId=str(member_row.id),
        userId=str(target_uuid),
        nickname=target_user.nickname if target_user else str(target_uuid),
        name=target_user.name if target_user else None,
        role=member_row.role,
        status=member_row.status,
        trustScore=float(target_user.trust_score) if target_user and target_user.trust_score is not None else 36.5,
        joinedAt=_format_datetime(member_row.joined_at),
        leftAt=None,
    )
