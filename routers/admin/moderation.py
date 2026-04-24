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

async def get_chat_moderation_logs(
    _: AdminContext = Depends(require_admin_moderation_permission),
    db: AsyncSession = Depends(get_db),
    party_id: str | None = Query(default=None),
    moderation_status: str | None = Query(default=None),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    keyword: str = Query(default=""),
    limit: int = Query(default=200, le=500),
):
    """탐지된 채팅 메시지 로그 (is_flagged=True)"""
    stmt = (
        select(PartyChat, Party, User)
        .join(Party, PartyChat.party_id == Party.id)
        .outerjoin(User, PartyChat.sender_id == User.id)
        .where(PartyChat.is_flagged == True)
        .order_by(PartyChat.created_at.desc())
        .limit(limit)
    )

    dt_from, dt_to = _date_range_bounds(date_from, date_to)
    if dt_from:
        stmt = stmt.where(PartyChat.created_at >= dt_from)
    if dt_to:
        stmt = stmt.where(PartyChat.created_at < dt_to)
    if party_id:
        try:
            import uuid as _uuid
            stmt = stmt.where(PartyChat.party_id == _uuid.UUID(party_id))
        except ValueError:
            pass
    if moderation_status:
        stmt = stmt.where(PartyChat.moderation_status == moderation_status)

    rows = (await db.execute(stmt)).all()
    q = keyword.lower().strip()

    result: list[ChatModerationLogOut] = []
    for chat, party, sender in rows:
        if q and q not in chat.message.lower() and q not in (sender.nickname if sender else "").lower():
            continue
        result.append(
            ChatModerationLogOut(
                id=str(chat.id),
                partyId=str(chat.party_id),
                partyTitle=party.title,
                senderId=str(chat.sender_id) if chat.sender_id else "-",
                senderNickname=sender.nickname if sender else "탈퇴/알 수 없음",
                message=chat.message,
                flagReason=chat.flag_reason,
                flagConfidence=chat.flag_confidence,
                flagStage=chat.flag_stage,
                moderationStatus=chat.moderation_status or "pending",
                isDeleted=chat.is_deleted,
                createdAt=_format_datetime(chat.created_at),
                warnCount=sender.chat_warn_count if sender else None,
            )
        )
    return result


@router.get("/moderation/chat-stats", response_model=ChatModerationStatsOut)
async def get_chat_moderation_stats(
    _: AdminContext = Depends(require_admin_moderation_permission),
    db: AsyncSession = Depends(get_db),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    """채팅 모더레이션 통계"""
    dt_from, dt_to = _date_range_bounds(date_from, date_to)

    base_flagged = select(func.count()).select_from(PartyChat).where(PartyChat.is_flagged == True)
    base_total = select(func.count()).select_from(PartyChat)

    if dt_from:
        base_flagged = base_flagged.where(PartyChat.created_at >= dt_from)
        base_total = base_total.where(PartyChat.created_at >= dt_from)
    if dt_to:
        base_flagged = base_flagged.where(PartyChat.created_at < dt_to)
        base_total = base_total.where(PartyChat.created_at < dt_to)

    total_flagged = (await db.scalar(base_flagged)) or 0
    total_messages = (await db.scalar(base_total)) or 0

    def _count_status(status: str):
        q = select(func.count()).select_from(PartyChat).where(
            PartyChat.is_flagged == True,
            PartyChat.moderation_status == status,
        )
        if dt_from:
            q = q.where(PartyChat.created_at >= dt_from)
        if dt_to:
            q = q.where(PartyChat.created_at < dt_to)
        return q

    blocked = (await db.scalar(_count_status("blocked"))) or 0
    warned = (await db.scalar(_count_status("warned"))) or 0
    false_positive = (await db.scalar(_count_status("false_positive"))) or 0
    pending = total_flagged - blocked - warned - false_positive

    detection_rate = round(total_flagged / total_messages * 100, 2) if total_messages > 0 else 0.0

    return ChatModerationStatsOut(
        totalFlagged=total_flagged,
        blocked=blocked,
        warned=warned,
        falsePositive=false_positive,
        pending=max(pending, 0),
        detectionRate=detection_rate,
    )


@router.patch("/moderation/chat-logs/{chat_id}/status")
async def update_chat_moderation_status(
    chat_id: str,
    status: str = Query(..., description="blocked / warned / false_positive / pending"),
    admin: AdminContext = Depends(require_admin_moderation_permission),
    db: AsyncSession = Depends(get_db),
):
    """탐지 오류 수정 (false_positive 마킹 등)"""
    if status not in {"blocked", "warned", "false_positive", "pending"}:
        raise HTTPException(status_code=400, detail="허용되지 않는 상태입니다.")
    try:
        import uuid as _uuid
        chat_uuid = _uuid.UUID(chat_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="유효하지 않은 ID입니다.")

    chat = await db.get(PartyChat, chat_uuid)
    if not chat:
        raise HTTPException(status_code=404, detail="채팅 메시지를 찾을 수 없습니다.")

    chat.moderation_status = status
    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="chat_moderation_status_updated",
        description=f"채팅 모더레이션 상태 변경 → {status}",
        target_id=chat_uuid,
    )
    await db.commit()
    return {"id": chat_id, "moderationStatus": status}

@router.get("/users/{user_id}/status-logs", response_model=list[UserStatusLogOut])
async def get_user_status_logs(
    user_id: str,
    _: AdminContext = Depends(require_admin_user_permission),
    db: AsyncSession = Depends(get_db),
):
    """특정 사용자의 상태변경(정상/주의/정지) 이력"""
    try:
        import uuid as _uuid
        user_uuid = _uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="유효하지 않은 사용자 ID입니다.")

    STATUS_ACTIONS = {"STATUS_정상", "STATUS_주의", "STATUS_정지"}

    rows = (
        await db.execute(
            select(ActivityLog, User)
            .outerjoin(User, ActivityLog.actor_user_id == User.id)
            .where(
                ActivityLog.target_id == user_uuid,
                ActivityLog.action_type.in_(STATUS_ACTIONS),
            )
            .order_by(ActivityLog.created_at.desc())
            .limit(100)
        )
    ).all()

    result: list[UserStatusLogOut] = []
    for log, actor in rows:
        to_status = log.action_type.replace("STATUS_", "")
        meta = log.extra_metadata or {}
        reason = meta.get("reason")

        # trigger 분류: reason에 "신고" 포함이면 report, actor가 없으면 auto, 나머지는 manual
        if not log.actor_user_id:
            trigger = "auto"
        elif reason and "신고" in reason:
            trigger = "report"
        else:
            trigger = "manual"

        result.append(
            UserStatusLogOut(
                id=str(log.id),
                toStatus=to_status,
                changedBy=_actor_display_name(actor),
                reason=reason,
                trigger=trigger,
                createdAt=_format_datetime(log.created_at),
            )
        )
    return result

# ── LSTM Shadow Mode 토글 ──────────────────────────────

@router.get("/captcha/shadow", tags=["admin-captcha"])
async def get_chat_moderation_trend(
    _: AdminContext = Depends(require_admin_moderation_permission),
    db: AsyncSession = Depends(get_db),
    period: str = Query(default="daily", pattern="^(daily|weekly|monthly)$"),
    start_date: str | None = Query(default=None),
    end_date: str | None = Query(default=None),
):
    from datetime import date as date_type
    e_date = date.today()
    if period == "daily":
        s_date = e_date - timedelta(days=6)
    elif period == "weekly":
        s_date = e_date - timedelta(weeks=8)
    else:
        s_date = e_date - timedelta(days=180)

    if start_date:
        s_date = date_type.fromisoformat(start_date)
    if end_date:
        e_date = date_type.fromisoformat(end_date)

    start_dt = datetime(s_date.year, s_date.month, s_date.day)
    end_dt = datetime(e_date.year, e_date.month, e_date.day) + timedelta(days=1)

    if period == "daily":
        trunc = "day"
    elif period == "weekly":
        trunc = "week"
    else:
        trunc = "month"

    result = await db.execute(text(f"""
        SELECT
            DATE_TRUNC('{trunc}', created_at + INTERVAL '9 hours')::date AS label,
            COUNT(*) FILTER (WHERE moderation_status = 'blocked') AS blocked,
            COUNT(*) FILTER (WHERE moderation_status = 'warned') AS warned,
            COUNT(*) FILTER (WHERE moderation_status = 'false_positive') AS false_positive,
            COUNT(*) AS total
        FROM party_chats
        WHERE is_flagged = TRUE
          AND created_at >= :start
          AND created_at < :end
        GROUP BY label
        ORDER BY label
    """), {"start": start_dt, "end": end_dt})

    rows = result.mappings().all()
    return [
        {
            "date": str(row["label"]),
            "blocked": row["blocked"] or 0,
            "warned": row["warned"] or 0,
            "false_positive": row["false_positive"] or 0,
            "total": row["total"] or 0,
        }
        for row in rows
    ]