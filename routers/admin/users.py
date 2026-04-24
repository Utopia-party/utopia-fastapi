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

@router.get("/users", response_model=list[AdminUserRecordOut])
async def get_admin_users(
    _: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    status_filter: str = Query(default="", alias="status"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    report_counts = _report_target_counts_subquery("user", "user_id")
    party_counts = (
        select(PartyMember.user_id.label("user_id"), func.count(PartyMember.id).label("count"))
        .group_by(PartyMember.user_id)
        .subquery()
    )
    latest_status_actions = _latest_user_status_actions_subquery()

    stmt = (
        select(
            User,
            func.coalesce(report_counts.c.count, 0),
            func.coalesce(party_counts.c.count, 0),
            latest_status_actions.c.action_type,
        )
        .outerjoin(report_counts, report_counts.c.user_id == User.id)
        .outerjoin(party_counts, party_counts.c.user_id == User.id)
        .outerjoin(latest_status_actions, latest_status_actions.c.target_user_id == User.id)
        .order_by(User.created_at.desc())
    )
    dt_from, dt_to = _date_range_bounds(date_from, date_to)
    if dt_from:
        stmt = stmt.where(User.created_at >= dt_from)
    if dt_to:
        stmt = stmt.where(User.created_at < dt_to)

    result = await db.execute(stmt)
    rows = result.all()

    items: list[AdminUserRecordOut] = []
    q = keyword.lower().strip()
    for user, report_count, party_count, manual_action in rows:
        status_label = _user_status_label(
            user,
            int(report_count),
            _manual_status_label(manual_action),
        )
        if status_filter and status_label != status_filter:
            continue
        if q and not (
            q in str(user.id).lower()
            or q in (user.name or "").lower()
            or q in (user.nickname or "").lower()
            or q in status_label.lower()
        ):
            continue

        items.append(
            AdminUserRecordOut(
                id=str(user.id),
                name=user.name,
                nickname=user.nickname,
                createdAt=_format_datetime(user.created_at),
                status=status_label,
                reportCount=int(report_count),
                partyCount=int(party_count),
                trustScore=float(user.trust_score) if user.trust_score is not None else 36.5,
                lastActive=_format_relative(user.last_login_at or user.updated_at),
            )
        )

    return items


@router.get("/users/{user_id}", response_model=AdminUserDetailOut)
async def get_admin_user_detail(
    user_id: str,
    _: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    report_count = await db.scalar(
        select(func.count())
        .select_from(Report)
        .where(func.lower(Report.target_type) == "user", Report.target_id == user.id)
    ) or 0
    party_count = await db.scalar(
        select(func.count()).select_from(PartyMember).where(PartyMember.user_id == user.id)
    ) or 0
    manual_action = await db.scalar(
        select(ActivityLog.action_type)
        .where(
            ActivityLog.target_id == user.id,
            ActivityLog.action_type.in_(["STATUS_정상", "STATUS_주의", "STATUS_정지"]),
        )
        .order_by(ActivityLog.created_at.desc())
        .limit(1)
    )
    recent_token_rows = (
        await db.execute(
            select(RefreshToken)
            .where(RefreshToken.user_id == user.id)
            .order_by(RefreshToken.created_at.desc())
            .limit(5)
        )
    ).scalars().all()
    trust_history_rows = (
        await db.execute(
            select(TrustScore)
            .where(TrustScore.user_id == user.id)
            .order_by(TrustScore.created_at.desc(), TrustScore.id.desc())
            .limit(10)
        )
    ).scalars().all()
    trust_creator_ids = {
        row.created_by for row in trust_history_rows if row.created_by is not None
    }
    trust_creators: dict[Any, User] = {}
    if trust_creator_ids:
        creator_rows = (
            await db.execute(select(User).where(User.id.in_(trust_creator_ids)))
        ).scalars().all()
        trust_creators = {creator.id: creator for creator in creator_rows}

    moderation_rows = (
        await db.execute(
            select(ModerationAction)
            .where(ModerationAction.user_id == user.id)
            .order_by(ModerationAction.created_at.desc())
            .limit(10)
        )
    ).scalars().all()
    moderation_actor_ids = {
        row.admin_id for row in moderation_rows if row.admin_id is not None
    }
    moderation_actors: dict[Any, User] = {}
    if moderation_actor_ids:
        actor_rows = (
            await db.execute(select(User).where(User.id.in_(moderation_actor_ids)))
        ).scalars().all()
        moderation_actors = {actor.id: actor for actor in actor_rows}

    return AdminUserDetailOut(
        id=str(user.id),
        email=user.email,
        nickname=user.nickname,
        name=user.name,
        phone=user.phone,
        role=user.role,
        status=_user_status_label(user, int(report_count), _manual_status_label(manual_action)),
        trustScore=float(user.trust_score) if user.trust_score is not None else 36.5,
        reportCount=int(report_count),
        partyCount=int(party_count),
        createdAt=_format_datetime(user.created_at),
        lastActive=_format_datetime(user.last_login_at or user.updated_at),
        bannedUntil=_format_datetime(user.banned_until) if user.banned_until else None,
        recentLoginIp=recent_token_rows[0].ip_address if recent_token_rows else None,
        recentLoginUserAgent=recent_token_rows[0].user_agent if recent_token_rows else None,
        recentLoginAt=_format_datetime(recent_token_rows[0].created_at) if recent_token_rows else None,
        trustHistories=[
            AdminUserTrustHistoryOut(
                id=str(row.id),
                title=row.reason,
                detail=_build_trust_history_detail(row),
                scoreChange=float(row.change_amount),
                trustScoreAfter=float(row.new_score),
                createdAt=_format_datetime(row.created_at),
                changedBy=_actor_display_name(trust_creators.get(row.created_by), "system"),
            )
            for row in trust_history_rows
        ],
        accessLogs=[
            AdminUserAccessLogOut(
                id=str(row.id),
                ipAddress=row.ip_address,
                userAgent=row.user_agent,
                createdAt=_format_datetime(row.created_at),
                isActive=row.revoked_at is None,
            )
            for row in recent_token_rows
        ],
        moderationHistories=[
            AdminModerationHistoryOut(
                id=str(row.id),
                actionType=_moderation_action_label(row.action_type),
                reason=row.reason,
                trustScoreChange=float(row.trust_score_change) if row.trust_score_change is not None else None,
                durationMinutes=row.duration_minutes,
                createdAt=_format_datetime(row.created_at),
                createdBy=_actor_display_name(moderation_actors.get(row.admin_id), "system"),
            )
            for row in moderation_rows
        ],
    )


@router.get("/services", response_model=list[AdminServiceRecordOut])
async def get_admin_services(
    _: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
):
    creator = aliased(User)
    result = await db.execute(
        select(Service, creator)
        .outerjoin(creator, Service.created_by == creator.id)
        .order_by(Service.updated_at.desc(), Service.created_at.desc())
    )
    rows = result.all()
    return [_serialize_admin_service(service, created_by) for service, created_by in rows]


@router.patch("/services/{service_id}", response_model=AdminServiceRecordOut)
async def update_admin_service(
    service_id: str,
    payload: AdminServiceUpdateIn,
    admin: AdminContext = Depends(require_admin_party_permission),
    db: AsyncSession = Depends(get_db),
):
    service = await db.get(Service, service_id)
    if not service:
        raise HTTPException(status_code=404, detail="서비스를 찾을 수 없습니다.")

    service.max_members = payload.maxMembers
    service.monthly_price = payload.monthlyPrice
    service.original_price = payload.originalPrice
    service.logo_image_key = payload.logoImageKey
    service.is_active = payload.isActive
    service.commission_rate = payload.commissionRate
    service.leader_discount_rate = payload.leaderDiscountRate
    service.referral_discount_rate = payload.referralDiscountRate

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="admin_service_updated",
        description=f"{service.name} 서비스 운영 값을 수정",
        path=f"/api/admin/services/{service_id}",
    )
    await _append_system_log(
        db,
        level="INFO",
        service="admin",
        message=f"서비스 설정 변경: {service.name}",
        actor=admin.user.nickname,
        admin_id=admin.user.id,
    )
    await db.commit()
    await db.refresh(service)

    created_by = await db.get(User, service.created_by) if service.created_by else None
    return _serialize_admin_service(service, created_by)


@router.patch("/users/{user_id}/status", response_model=AdminUserRecordOut)
async def update_admin_user_status(
    user_id: str,
    payload: AdminUserStatusUpdateIn,
    admin: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
):
    target_user = await db.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    if payload.status not in {"정상", "주의", "정지"}:
        raise HTTPException(status_code=400, detail="허용되지 않은 상태입니다.")

    if payload.status == "정지" and target_user.id == admin.user.id:
        raise HTTPException(status_code=400, detail="본인 계정은 정지할 수 없습니다.")

    if payload.status == "정지":
        target_role = await db.scalar(select(AdminRole).where(AdminRole.user_id == target_user.id))
        if target_role and target_role.can_manage_admins:
            if await _count_root_admins(db) <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="마지막 ROOT 관리자는 정지할 수 없습니다.",
                )
        target_user.is_active = False
        target_user.banned_until = datetime.now(timezone.utc) + timedelta(days=30)
    else:
        target_user.is_active = True
        target_user.banned_until = None

    db.add(
        Notification(
            user_id=target_user.id,
            type="SYSTEM",
            title="계정 상태 변경",
            message=f"관리자에 의해 계정 상태가 '{payload.status}'로 변경되었습니다.",
        )
    )
    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type=f"STATUS_{payload.status}",
        description=f"{target_user.nickname} 상태를 {payload.status}로 변경",
        path=f"/api/admin/users/{user_id}/status",
        target_id=target_user.id,
        reason=payload.reason,
    )
    await _append_system_log(
        db,
        level="INFO",
        service="admin",
        message=f"사용자 상태 변경: {target_user.nickname} -> {payload.status}",
        actor=admin.user.nickname,
        admin_id=admin.user.id,
    )
    await db.commit()
    await db.refresh(target_user)

    report_count = await db.scalar(
        select(func.count())
        .select_from(Report)
        .where(func.lower(Report.target_type) == "user", Report.target_id == target_user.id)
    ) or 0
    party_count = await db.scalar(
        select(func.count()).select_from(PartyMember).where(PartyMember.user_id == target_user.id)
    ) or 0

    return AdminUserRecordOut(
        id=str(target_user.id),
        name=target_user.name,
        nickname=target_user.nickname,
        createdAt=_format_datetime(target_user.created_at),
        status=_user_status_label(target_user, int(report_count), payload.status),
        reportCount=int(report_count),
        partyCount=int(party_count),
        trustScore=float(target_user.trust_score) if target_user.trust_score is not None else 36.5,
        lastActive=_format_relative(target_user.last_login_at or target_user.updated_at),
    )


@router.get("/users/{user_id}/status-logs", response_model=list[AdminUserStatusLogOut])
async def get_admin_user_status_logs(
    user_id: str,
    _: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
):
    target_user = await db.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    rows = (
        await db.execute(
            select(ActivityLog)
            .where(
                ActivityLog.target_id == target_user.id,
                ActivityLog.action_type.in_(["STATUS_정상", "STATUS_주의", "STATUS_정지"]),
            )
            .order_by(ActivityLog.created_at.desc())
            .limit(50)
        )
    ).scalars().all()

    actor_ids = {row.actor_user_id for row in rows if row.actor_user_id is not None}
    actors: dict[Any, User] = {}
    if actor_ids:
        users = (
            await db.execute(select(User).where(User.id.in_(actor_ids)))
        ).scalars().all()
        actors = {user.id: user for user in users}

    return [
        AdminUserStatusLogOut(
            id=str(row.id),
            toStatus=row.action_type.replace("STATUS_", ""),
            changedBy=_actor_display_name(actors.get(row.actor_user_id), "system"),
            reason=row.description,
            trigger="manual",
            createdAt=_format_datetime(row.created_at),
        )
        for row in rows
    ]


@router.patch("/users/{user_id}/trust-score", response_model=AdminUserDetailOut)
async def update_admin_user_trust_score(
    user_id: str,
    payload: AdminUserTrustScoreUpdateIn,
    admin: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
):
    target_user = await db.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    next_score = round(float(payload.trustScore), 1)
    if next_score < 0 or next_score > 100:
        raise HTTPException(status_code=400, detail="신뢰도는 0점 이상 100점 이하여야 합니다.")

    previous_score = float(target_user.trust_score) if target_user.trust_score is not None else 36.5
    target_user.trust_score = next_score

    trust_row = TrustScore(
        user_id=target_user.id,
        previous_score=previous_score,
        new_score=next_score,
        change_amount=round(next_score - previous_score, 1),
        reason=(payload.reason or "관리자 수동 조정").strip() or "관리자 수동 조정",
        created_by=admin.user.id,
    )
    db.add(trust_row)

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="TRUST_SCORE_UPDATED",
        description=(
            f"{target_user.nickname} 신뢰도를 {previous_score:.1f} → {next_score:.1f}로 변경"
            f"{f' ({payload.reason.strip()})' if payload.reason and payload.reason.strip() else ''}"
        ),
        path=f"/api/admin/users/{user_id}/trust-score",
        target_id=target_user.id,
    )
    await _append_system_log(
        db,
        level="INFO",
        service="admin",
        message=f"사용자 신뢰도 변경: {target_user.nickname} {previous_score:.1f} → {next_score:.1f}",
        actor=admin.user.nickname,
        admin_id=admin.user.id,
    )
    await db.commit()

    return await get_admin_user_detail(user_id, admin, db)
