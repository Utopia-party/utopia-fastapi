from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
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

router = APIRouter(prefix="/admin", tags=["admin"])


@dataclass
class AdminContext:
    user: User
    role: AdminRole


def _format_datetime(value: datetime | None) -> str:
    if not value:
        return "-"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _format_relative(value: datetime | None) -> str:
    if not value:
        return "-"

    delta = datetime.now(timezone.utc) - value.astimezone(timezone.utc)
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "방금"
    if seconds < 3600:
        return f"{seconds // 60}분 전"
    if seconds < 86400:
        return f"{seconds // 3600}시간 전"
    return f"{seconds // 86400}일 전"


def _to_int(value: Decimal | int | float | None) -> int:
    if value is None:
        return 0
    if isinstance(value, Decimal):
        return int(float(value))
    return int(value)


def _utc_day_start(value: date) -> datetime:
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def _date_range_bounds(
    date_from: date | None,
    date_to: date | None,
) -> tuple[datetime | None, datetime | None]:
    dt_from = _utc_day_start(date_from) if date_from else None
    dt_to = (_utc_day_start(date_to) + timedelta(days=1)) if date_to else None
    return dt_from, dt_to


def _format_change(current: int | float, comparison: int | float, suffix: str = "%") -> tuple[str, str]:
    if comparison == 0:
        if current == 0:
            return "변동 없음", "flat"
        return "비교 기준 없음", "up"

    change_rate = ((current - comparison) / comparison) * 100
    if abs(change_rate) < 0.05:
        return "변동 없음", "flat"
    direction = "up" if change_rate > 0 else "down"
    return f"{change_rate:+.1f}{suffix}", direction


def _bucket_labels(
    start_date: date,
    end_date: date,
) -> tuple[list[date], str]:
    total_days = (end_date - start_date).days + 1
    if total_days <= 31:
        return [start_date + timedelta(days=offset) for offset in range(total_days)], "day"

    month_starts: list[date] = []
    cursor = date(start_date.year, start_date.month, 1)
    last = date(end_date.year, end_date.month, 1)
    while cursor <= last:
        month_starts.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return month_starts, "month"


def _shift_comparison_range(
    start_date: date,
    end_date: date,
    compare_mode: str,
) -> tuple[date, date]:
    if compare_mode == "year_over_year":
        def shift_one_year(value: date) -> date:
            try:
                return date(value.year - 1, value.month, value.day)
            except ValueError:
                if value.month == 2 and value.day == 29:
                    return date(value.year - 1, 2, 28)
                raise

        return shift_one_year(start_date), shift_one_year(end_date)

    span_days = (end_date - start_date).days + 1
    comparison_end = start_date - timedelta(days=1)
    comparison_start = comparison_end - timedelta(days=span_days - 1)
    return comparison_start, comparison_end


def _series_label(value: date, bucket_mode: str) -> str:
    if bucket_mode == "day":
        return value.strftime("%m/%d")
    return value.strftime("%Y-%m")


def _user_display_name(user: User | None) -> str:
    if not user:
        return "-"
    return user.name or user.nickname or str(user.id)


def _actor_display_name(user: User | None, fallback: str | None = None) -> str:
    if user:
        return user.nickname or user.name or str(user.id)
    return fallback or "system"


def _build_trust_history_detail(row: TrustScore) -> str | None:
    parts: list[str] = []
    if row.previous_score is not None and row.new_score is not None:
        parts.append(f"{float(row.previous_score):.1f} → {float(row.new_score):.1f}")
    if row.reference_id:
        parts.append(f"reference_id: {row.reference_id}")
    if not parts:
        return None
    return " | ".join(parts)


def _moderation_action_label(value: str | None) -> str:
    mapping = {
        "warn": "경고",
        "warning": "경고",
        "ban": "차단",
        "block": "차단",
        "suspend": "정지",
        "mute": "채팅 제한",
        "review": "검토",
    }
    normalized = (value or "").strip().lower()
    return mapping.get(normalized, value or "-")


def _admin_permissions_for_role(role: str) -> dict[str, Any]:
    role = role.upper()
    if role == "ROOT":
        return {
            "can_manage_users": True,
            "can_manage_parties": True,
            "can_manage_reports": True,
            "can_manage_moderation": True,
            "can_approve_receipts": True,
            "can_approve_settlements": True,
            "can_view_logs": True,
            "can_manage_admins": True,
        }
    return {
        "can_manage_users": True,
        "can_manage_parties": True,
        "can_manage_reports": True,
        "can_manage_moderation": True,
        "can_approve_receipts": True,
        "can_approve_settlements": True,
        "can_view_logs": True,
        "can_manage_admins": False,
    }


def _manual_status_label(action_type: str | None) -> str | None:
    if not action_type:
        return None
    return {
        "STATUS_정상": "정상",
        "STATUS_주의": "주의",
        "STATUS_정지": "정지",
    }.get(action_type.upper())


def _user_status_label(user: User, report_count: int, manual_status: str | None = None) -> str:
    if not user.is_active:
        return "정지"
    if manual_status in {"정상", "주의"}:
        return manual_status
    trust_score = float(user.trust_score) if user.trust_score is not None else 36.5
    if trust_score < 36.5 or report_count >= 2:
        return "주의"
    return "정상"


def _party_status_label(party: Party, report_count: int) -> str:
    # 파티 종료 수정
    if party.status.lower() == "ended":
        return "종료됨"
    # 파티 종료 수정
    if report_count > 0:
        return "위험"
    if party.status.lower() == "recruiting":
        return "모집중"
    return "운영중"


def _report_status_label(value: str) -> str:
    normalized = (value or "").strip()
    upper = normalized.upper()
    lower = normalized.lower()

    return {
        "PENDING": "접수",
        "IN_REVIEW": "검토중",
        "APPROVED": "처리",
        "REJECTED": "기각",
        # 이전 로컬 상태값도 읽을 수 있게 유지
        "pending": "접수",
        "processed": "처리",
        "approved": "처리",
        "rejected": "기각",
        "in_review": "검토중",
        "appealed": "검토중",
        "auto_processed": "처리",
    }.get(upper if upper in {"PENDING", "IN_REVIEW", "APPROVED", "REJECTED"} else lower, value)


def _report_status_code(value: str) -> str:
    normalized = (value or "").strip()

    return {
        "접수": "PENDING",
        "검토중": "IN_REVIEW",
        "처리": "APPROVED",
        "기각": "REJECTED",
        "PENDING": "PENDING",
        "IN_REVIEW": "IN_REVIEW",
        "APPROVED": "APPROVED",
        "REJECTED": "REJECTED",
        "pending": "PENDING",
        "in_review": "IN_REVIEW",
        "approved": "APPROVED",
        "rejected": "REJECTED",
        # 이전 로컬 상태값도 서버 상태 체계로 정규화
        "processed": "APPROVED",
        "appealed": "IN_REVIEW",
        "auto_processed": "APPROVED",
    }.get(normalized, normalized.upper())


def _report_type_label(value: str) -> str:
    return {
        "user": "사용자",
        "party": "파티",
        "chat": "채팅",
    }.get(value.lower(), value)


def _report_target_counts_subquery(target_type: str, label: str):
    return (
        select(Report.target_id.label(label), func.count(Report.id).label("count"))
        .where(func.lower(Report.target_type) == target_type)
        .group_by(Report.target_id)
        .subquery()
    )


def _receipt_status_label(value: str) -> str:
    return {
        "pending": "대기",
        "approved": "승인",
        "rejected": "거절",
    }.get(value.lower(), value)


def _receipt_status_code(value: str) -> str:
    return {
        "대기": "pending",
        "승인": "approved",
        "거절": "rejected",
    }.get(value, value.lower())


def _settlement_status_label(value: str) -> str:
    return {
        "pending": "대기",
        "approved": "승인",
        "rejected": "거절",
    }.get(value.lower(), value)


def _settlement_status_code(value: str) -> str:
    return {
        "대기": "pending",
        "승인": "approved",
        "거절": "rejected",
    }.get(value, value.lower())


async def _append_activity_log(
    db: AsyncSession,
    *,
    actor_user_id: Any | None,
    action_type: str,
    description: str,
    path: str | None = None,
    ip_address: str | None = None,
    target_id: Any | None = None,
    reason: str | None = None,
) -> None:
    metadata: dict = {}
    if path:
        metadata["path"] = path
    if reason:
        metadata["reason"] = reason
    db.add(
        ActivityLog(
            actor_user_id=actor_user_id,
            action_type=action_type,
            description=description,
            ip_address=ip_address,
            extra_metadata=metadata or None,
            target_id=target_id,
        )
    )


async def _append_system_log(
    db: AsyncSession,
    *,
    level: str,
    service: str,
    message: str,
    actor: str | None = None,
    admin_id: Any | None = None,
) -> None:
    metadata = {"actor": actor} if actor else None
    db.add(
        SystemLog(
            level=level,
            service=service,
            message=message,
            extra_metadata=metadata,
            admin_id=admin_id,
        )
    )


def _admin_permissions_payload(payload: AdminRoleUpdateIn) -> dict[str, bool]:
    return {
        "can_manage_users": payload.canManageUsers,
        "can_manage_parties": payload.canManageParties,
        "can_manage_reports": payload.canManageReports,
        "can_manage_moderation": payload.canManageChatModeration,
        "can_approve_receipts": payload.canManageCaptcha,
        "can_approve_settlements": payload.canApproveSettlements,
        "can_view_logs": payload.canViewLogs,
        "can_manage_admins": payload.canManageAdmins,
    }


def _has_any_admin_permission(values: dict[str, bool]) -> bool:
    return any(values.values())


def _serialize_admin_permissions(role: AdminRole) -> AdminPermissionOut:
    return AdminPermissionOut(
        canManageUsers=role.can_manage_users,
        canManageParties=role.can_manage_parties,
        canManageReports=role.can_manage_reports,
        canManageChatModeration=role.can_manage_moderation,
        canManageCaptcha=role.can_approve_receipts,
        canApproveSettlements=role.can_approve_settlements,
        canViewLogs=role.can_view_logs,
        canManageAdmins=role.can_manage_admins,
    )


def _serialize_admin_role(role: AdminRole, user: User, created_by: User | None) -> AdminRoleRecordOut:
    return AdminRoleRecordOut(
        id=str(role.id),
        userId=str(user.id),
        adminId=user.nickname or user.email,
        canManageUsers=role.can_manage_users,
        canManageParties=role.can_manage_parties,
        canManageReports=role.can_manage_reports,
        canManageChatModeration=role.can_manage_moderation,
        canManageCaptcha=role.can_approve_receipts,
        canApproveSettlements=role.can_approve_settlements,
        canViewLogs=role.can_view_logs,
        canManageAdmins=role.can_manage_admins,
        lastUpdated=_format_datetime(role.updated_at),
        updatedBy=(created_by.nickname or created_by.email) if created_by else "system",
    )


def _serialize_admin_service(service: Service, created_by: User | None) -> AdminServiceRecordOut:
    return AdminServiceRecordOut(
        id=str(service.id),
        name=service.name,
        category=service.category,
        maxMembers=service.max_members,
        monthlyPrice=service.monthly_price,
        originalPrice=(
            service.original_price
            if service.original_price is not None
            else service.monthly_price
        ),
        logoImageKey=service.logo_image_key,
        logoImageUrl=build_minio_asset_url(service.logo_image_key),
        isActive=service.is_active,
        createdBy=(created_by.nickname or created_by.email) if created_by else "-",
        createdAt=_format_datetime(service.created_at),
        updatedAt=_format_datetime(service.updated_at),
        commissionRate=float(service.commission_rate or 0),
        leaderDiscountRate=float(service.leader_discount_rate or 0),
        referralDiscountRate=float(service.referral_discount_rate or 0),
    )


async def _report_target_display_map(
    db: AsyncSession,
    reports: list[Report],
) -> dict[tuple[str, Any], str]:
    display_map: dict[tuple[str, Any], str] = {}
    user_ids = {report.target_id for report in reports if report.target_type.lower() == "user"}
    party_ids = {report.target_id for report in reports if report.target_type.lower() in {"party", "chat"}}
    chat_ids = {report.target_id for report in reports if report.target_type.lower() == "chat"}

    users_by_id: dict[Any, User] = {}
    parties_by_id: dict[Any, Party] = {}
    chats_by_id: dict[Any, PartyChat] = {}

    if user_ids:
        user_rows = (await db.execute(select(User).where(User.id.in_(user_ids)))).scalars().all()
        users_by_id = {user.id: user for user in user_rows}
    if party_ids:
        party_rows = (await db.execute(select(Party).where(Party.id.in_(party_ids)))).scalars().all()
        parties_by_id = {party.id: party for party in party_rows}
    if chat_ids:
        chat_rows = (await db.execute(select(PartyChat).where(PartyChat.id.in_(chat_ids)))).scalars().all()
        chats_by_id = {chat.id: chat for chat in chat_rows}
        sender_ids = {chat.sender_id for chat in chat_rows if chat.sender_id is not None}
        missing_user_ids = sender_ids - set(users_by_id.keys())
        if missing_user_ids:
            sender_rows = (
                await db.execute(select(User).where(User.id.in_(missing_user_ids)))
            ).scalars().all()
            users_by_id.update({user.id: user for user in sender_rows})

    for report in reports:
        target_type = report.target_type.lower()
        display_name = report.target_snapshot_name

        if target_type == "user":
            target_user = users_by_id.get(report.target_id)
            display_name = display_name or _user_display_name(target_user)
        elif target_type == "party":
            target_party = parties_by_id.get(report.target_id)
            display_name = display_name or (target_party.title if target_party else None)
        elif target_type == "chat":
            target_chat = chats_by_id.get(report.target_id)
            if target_chat:
                target_party = parties_by_id.get(target_chat.party_id)
                sender = users_by_id.get(target_chat.sender_id)
                chat_label = sender.nickname if sender else "채팅 사용자"
                party_label = target_party.title if target_party else "파티 채팅"
                display_name = display_name or f"{party_label} / {chat_label}"

        display_map[(target_type, report.target_id)] = display_name or str(report.target_id)

    return display_map


def _assert_admin_permission(
    admin: AdminContext,
    permission_name: str,
    detail: str,
) -> AdminContext:
    if not getattr(admin.role, permission_name):
        raise HTTPException(status_code=403, detail=detail)
    return admin


def _latest_user_status_actions_subquery():
    ranked_actions = (
        select(
            ActivityLog.target_id.label("target_user_id"),
            ActivityLog.action_type.label("action_type"),
            func.row_number()
            .over(
                partition_by=ActivityLog.target_id,
                order_by=ActivityLog.created_at.desc(),
            )
            .label("row_num"),
        )
        .where(
            ActivityLog.target_id.is_not(None),
            ActivityLog.action_type.in_(["STATUS_정상", "STATUS_주의", "STATUS_정지"]),
        )
        .subquery()
    )
    return (
        select(ranked_actions.c.target_user_id, ranked_actions.c.action_type)
        .where(ranked_actions.c.row_num == 1)
        .subquery()
    )


async def _count_root_admins(db: AsyncSession) -> int:
    return int(
        await db.scalar(
            select(func.count()).select_from(AdminRole).where(AdminRole.can_manage_admins.is_(True))
        )
        or 0
    )


async def _ensure_admin_role(db: AsyncSession, user: User) -> AdminRole:
    result = await db.execute(select(AdminRole).where(AdminRole.user_id == user.id))
    role = result.scalar_one_or_none()
    if role:
        if role.created_by is None:
            role.created_by = user.id
            await db.commit()
            await db.refresh(role)
        return role

    existing_count = await db.scalar(select(func.count()).select_from(AdminRole)) or 0
    defaults = _admin_permissions_for_role("ROOT" if existing_count == 0 else "ADMIN")
    role = AdminRole(
        user_id=user.id,
        created_by=user.id,
        **defaults,
    )
    db.add(role)
    await db.commit()
    await db.refresh(role)
    return role


async def require_admin_context(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
) -> AdminContext:
    if (current_user.role or "").lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자만 접근할 수 있습니다.",
        )
    role = await _ensure_admin_role(db, current_user)
    return AdminContext(user=current_user, role=role)


async def require_admin_user_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_users", "사용자 관리 권한이 없습니다.")


async def require_admin_party_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_parties", "파티 관리 권한이 없습니다.")


async def require_admin_report_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_reports", "신고 관리 권한이 없습니다.")


async def require_admin_receipt_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_approve_receipts", "영수증 승인 권한이 없습니다.")


async def require_admin_settlement_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_approve_settlements", "정산 승인 권한이 없습니다.")


async def require_admin_log_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_view_logs", "시스템 로그 조회 권한이 없습니다.")


async def require_admin_moderation_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_moderation", "모더레이션 관리 권한이 없습니다.")


async def require_admin_role_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_admins", "관리자 권한 변경 권한이 없습니다.")


@router.get("/dashboard", response_model=AdminDashboardOut)
async def get_admin_dashboard(
    _: AdminContext = Depends(require_admin_context),
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

    total_users = await db.scalar(select(func.count()).select_from(User)) or 0
    active_users = await db.scalar(
        select(func.count()).select_from(User).where(User.is_active.is_(True))
    ) or 0
    suspended_users = await db.scalar(
        select(func.count()).select_from(User).where(User.is_active.is_(False))
    ) or 0
    admin_users = await db.scalar(
        select(func.count()).select_from(User).where(func.lower(User.role) == "admin")
    ) or 0

    current_signups = len(current_users)
    comparison_signups = len(comparison_users)
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
    suspended_users_count = int(suspended_users)

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
            {"label": "전체 회원", "value": f"{total_users:,}"},
            {"label": "활성 사용자", "value": f"{active_users:,}"},
            {"label": "정지 사용자", "value": f"{suspended_users:,}"},
            {"label": "관리자 계정", "value": f"{admin_users:,}"},
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


@router.get("/me", response_model=AdminPermissionOut)
async def get_admin_me(
    admin: AdminContext = Depends(require_admin_context),
):
    return _serialize_admin_permissions(admin.role)


@router.get("/roles", response_model=list[AdminRoleRecordOut])
async def get_admin_roles(
    _: AdminContext = Depends(require_admin_role_permission),
    db: AsyncSession = Depends(get_db),
):
    creator = aliased(User)
    result = await db.execute(
        select(AdminRole, User, creator)
        .join(User, AdminRole.user_id == User.id)
        .outerjoin(creator, AdminRole.created_by == creator.id)
        .order_by(AdminRole.updated_at.desc())
    )
    rows = result.all()

    return [_serialize_admin_role(role, user, created_by) for role, user, created_by in rows]


@router.put("/roles/{user_id}", response_model=AdminRoleRecordOut)
async def update_admin_role(
    user_id: str,
    payload: AdminRoleUpdateIn,
    admin: AdminContext = Depends(require_admin_role_permission),
    db: AsyncSession = Depends(get_db),
):
    target_user = await db.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="관리자 대상으로 지정한 사용자가 없습니다.")

    next_permissions = _admin_permissions_payload(payload)
    if not _has_any_admin_permission(next_permissions):
        raise HTTPException(status_code=400, detail="최소 하나 이상의 관리자 권한이 필요합니다.")

    result = await db.execute(select(AdminRole).where(AdminRole.user_id == target_user.id))
    role_row = result.scalar_one_or_none()

    if target_user.id == admin.user.id:
        raise HTTPException(
            status_code=400,
            detail="본인 관리자 권한은 직접 변경할 수 없습니다.",
        )

    if role_row and role_row.can_manage_admins and not next_permissions["can_manage_admins"]:
        if await _count_root_admins(db) <= 1:
            raise HTTPException(
                status_code=400,
                detail="마지막 ROOT 관리자는 권한을 변경할 수 없습니다.",
            )

    target_user.role = "admin"
    if role_row is None:
        role_row = AdminRole(
            user_id=target_user.id,
            created_by=admin.user.id,
            **next_permissions,
        )
        db.add(role_row)
    else:
        for key, value in next_permissions.items():
            setattr(role_row, key, value)
        if role_row.created_by is None:
            role_row.created_by = admin.user.id

    await _append_activity_log(
        db,
        actor_user_id=admin.user.id,
        action_type="admin_role_updated",
        description=f"{target_user.nickname} 관리자 권한 세트를 변경",
        path=f"/api/admin/roles/{user_id}",
    )
    await _append_system_log(
        db,
        level="INFO",
        service="admin",
        message=f"관리자 권한 변경: {target_user.nickname}",
        actor=admin.user.nickname,
        admin_id=admin.user.id,
    )
    await db.commit()
    await db.refresh(role_row)

    return _serialize_admin_role(role_row, target_user, admin.user)


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
                leaderId=user.nickname,
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
        leaderId=host.nickname if host else str(party.leader_id),
        memberCount=party.current_members,
        status=_party_status_label(party, int(report_count)),
        reportCount=int(report_count),
        monthlyAmount=party.monthly_per_person * party.current_members,
        lastPayment="종료됨",
    )


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


@router.get("/logs", response_model=list[SystemLogRecordOut])
async def get_admin_logs(
    _: AdminContext = Depends(require_admin_log_permission),
    db: AsyncSession = Depends(get_db),
    keyword: str = Query(default=""),
    log_type: str = Query(default="", alias="type"),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
):
    logs: list[SystemLogRecordOut] = []

    dt_from, dt_to = _date_range_bounds(date_from, date_to)

    activity_stmt = select(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(100)
    system_stmt = select(SystemLog).order_by(SystemLog.created_at.desc()).limit(100)
    moderation_stmt = select(ModerationAction).order_by(ModerationAction.created_at.desc()).limit(100)

    if dt_from:
        activity_stmt = activity_stmt.where(ActivityLog.created_at >= dt_from)
        system_stmt = system_stmt.where(SystemLog.created_at >= dt_from)
        moderation_stmt = moderation_stmt.where(ModerationAction.created_at >= dt_from)
    if dt_to:
        activity_stmt = activity_stmt.where(ActivityLog.created_at < dt_to)
        system_stmt = system_stmt.where(SystemLog.created_at < dt_to)
        moderation_stmt = moderation_stmt.where(ModerationAction.created_at < dt_to)

    activity_rows = (await db.execute(activity_stmt)).scalars().all()
    system_rows = (await db.execute(system_stmt)).scalars().all()
    moderation_rows = (await db.execute(moderation_stmt)).scalars().all()

    actor_ids = {
        row.actor_user_id
        for row in activity_rows
        if row.actor_user_id is not None
    }
    actor_ids.update(row.admin_id for row in system_rows if row.admin_id is not None)
    actor_ids.update(row.admin_id for row in moderation_rows if row.admin_id is not None)

    users_by_id: dict[Any, User] = {}
    if actor_ids:
        actor_users = (
            await db.execute(select(User).where(User.id.in_(actor_ids)))
        ).scalars().all()
        users_by_id = {user.id: user for user in actor_users}

    logs.extend(
        [
            SystemLogRecordOut(
                id=str(row.id),
                timestamp=_format_datetime(row.created_at),
                type="ADMIN_ACTION",
                message=row.description,
                actor=_actor_display_name(
                    users_by_id.get(row.actor_user_id),
                    "system",
                ),
            )
            for row in activity_rows
        ]
    )
    logs.extend(
        [
            SystemLogRecordOut(
                id=str(row.id),
                timestamp=_format_datetime(row.created_at),
                type=row.level.upper(),
                message=row.message,
                actor=_actor_display_name(
                    users_by_id.get(row.admin_id),
                    ((row.extra_metadata or {}).get("actor") if row.extra_metadata else None)
                    or row.service,
                ),
            )
            for row in system_rows
        ]
    )
    logs.extend(
        [
            SystemLogRecordOut(
                id=str(row.id),
                timestamp=_format_datetime(row.created_at),
                type="ADMIN_ACTION",
                message=f"{row.action_type}: {row.reason or '-'}",
                actor=_actor_display_name(
                    users_by_id.get(row.admin_id),
                    "system",
                ),
            )
            for row in moderation_rows
        ]
    )

    logs.sort(key=lambda item: item.timestamp, reverse=True)

    q = keyword.lower().strip()
    lt = log_type.upper().strip()
    if q or lt:
        filtered: list[SystemLogRecordOut] = []
        for log in logs:
            if lt and log.type.upper() != lt:
                continue
            if q and not (
                q in log.message.lower()
                or q in log.actor.lower()
                or q in log.type.lower()
            ):
                continue
            filtered.append(log)
        return filtered[:200]

    return logs[:200]

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
    member_row.left_at = datetime.now(timezone.utc)
    if party.current_members and party.current_members > 0:
        party.current_members -= 1

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

    # 강퇴 알림
    if target_user:
        db.add(
            Notification(
                user_id=target_uuid,
                type="PARTY",
                title="파티 강퇴 안내",
                message=f"관리자에 의해 '{party.title}' 파티에서 강퇴되었습니다. 사유: {payload.reason or '운영 정책 위반'}",
                reference_type="party",
                reference_id=party.id,
            )
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

@router.get("/moderation/chat-logs", response_model=list[ChatModerationLogOut])
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
                moderationStatus=chat.moderation_status or "pending",
                isDeleted=chat.is_deleted,
                createdAt=_format_datetime(chat.created_at),
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
async def get_shadow_mode(current_user: User = Depends(require_user)):
    """현재 LSTM Shadow Mode 상태 조회"""
    return {
        "shadow_mode": settings.LSTM_SHADOW_MODE,
        "lstm_weight": settings.LSTM_WEIGHT,
        "score_formula": (
            "rule × {r:.0%} + KNN × {k:.0%} + LSTM × {l:.0%}".format(
                r=1.0 - settings.LSTM_WEIGHT - 0.2,
                k=0.2,
                l=settings.LSTM_WEIGHT,
            )
            if not settings.LSTM_SHADOW_MODE
            else "rule × (1-knn_w) + KNN × knn_w  (LSTM 로그만)"
        ),
    }


@router.put("/captcha/shadow", tags=["admin-captcha"])
async def toggle_shadow_mode(current_user: User = Depends(require_user)):
    """LSTM Shadow Mode ON/OFF 토글 (런타임 변경)"""
    settings.LSTM_SHADOW_MODE = not settings.LSTM_SHADOW_MODE
    new_state = settings.LSTM_SHADOW_MODE

    return {
        "shadow_mode": new_state,
        "message": (
            "LSTM Shadow ON — LSTM은 로그만 기록, final_score에 미반영"
            if new_state
            else "LSTM Shadow OFF — LSTM이 final_score에 반영됨 "
                 f"(rule×{1.0 - settings.LSTM_WEIGHT - 0.2:.0%} + KNN×20% + LSTM×{settings.LSTM_WEIGHT:.0%})"
        ),
    }


# ── IP 제재 관리 ──────────────────────────────────────

_CAPTCHA_KEY_PREFIXES = [
    "captcha:lock:",
    "captcha:lock-count:",
    "captcha:ban:",
    "captcha:wait:",
    "captcha:force-challenge:",
]


@router.get("/captcha/blocked-ips", tags=["admin-captcha"])
async def list_blocked_ips(current_user: User = Depends(require_user)):
    """현재 잠금/밴 상태인 IP 목록 조회"""
    blocked: dict[str, dict] = {}

    for prefix in _CAPTCHA_KEY_PREFIXES:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=f"{prefix}*", count=100)
            for key in keys:
                key_str = key if isinstance(key, str) else key.decode()
                ip = key_str.replace(prefix, "")
                if ip not in blocked:
                    blocked[ip] = {"ip": ip, "lock": False, "ban": False, "wait": False, "lock_count": 0, "ttl": {}}

                ttl = await redis_client.ttl(key_str)

                if prefix == "captcha:lock:":
                    blocked[ip]["lock"] = True
                    blocked[ip]["ttl"]["lock"] = ttl
                elif prefix == "captcha:ban:":
                    blocked[ip]["ban"] = True
                    blocked[ip]["ttl"]["ban"] = ttl
                elif prefix == "captcha:wait:":
                    blocked[ip]["wait"] = True
                    blocked[ip]["ttl"]["wait"] = ttl
                elif prefix == "captcha:lock-count:":
                    val = await redis_client.get(key_str)
                    blocked[ip]["lock_count"] = int(val) if val else 0

            if cursor == 0:
                break

    # ban > lock > wait 우선순위로 정렬
    items = sorted(
        blocked.values(),
        key=lambda x: (x["ban"], x["lock"], x["wait"]),
        reverse=True,
    )
    return {"blocked_ips": items, "total": len(items)}


@router.delete("/captcha/blocked-ips/{ip}", tags=["admin-captcha"])
async def unblock_ip(ip: str, current_user: User = Depends(require_user)):
    """특정 IP의 모든 캡챠 제재 해제"""
    deleted_keys = []
    for prefix in _CAPTCHA_KEY_PREFIXES:
        key = f"{prefix}{ip}"
        result = await redis_client.delete(key)
        if result:
            deleted_keys.append(key)

    return {
        "ip": ip,
        "unblocked": len(deleted_keys) > 0,
        "deleted_keys": deleted_keys,
        "message": f"{ip} 제재 해제 완료" if deleted_keys else f"{ip}에 대한 제재가 없습니다",
    }


@router.delete("/captcha/blocked-ips", tags=["admin-captcha"])
async def unblock_all_ips(current_user: User = Depends(require_user)):
    """모든 IP의 캡챠 제재 해제 (FLUSHDB 대신 캡챠 키만 삭제)"""
    total_deleted = 0
    for prefix in _CAPTCHA_KEY_PREFIXES:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=f"{prefix}*", count=100)
            if keys:
                await redis_client.delete(*keys)
                total_deleted += len(keys)
            if cursor == 0:
                break

    return {
        "total_deleted": total_deleted,
        "message": f"캡챠 제재 {total_deleted}건 전체 해제 완료",
    }


# ── 캡챠 수치 설정 (런타임) ──────────────────────────────

@router.get("/captcha/config", tags=["admin-captcha"])
async def get_captcha_config(current_user: User = Depends(require_user)):
    """현재 캡챠 가중치 및 임계값 조회"""
    lstm_w = getattr(settings, "LSTM_WEIGHT", 0.7)
    knn_w = getattr(settings, "KNN_WEIGHT", 0.2)
    rule_w = round(1.0 - lstm_w - knn_w, 4)
    return {
        "lstm_weight": lstm_w,
        "knn_weight": knn_w,
        "rule_weight": rule_w,
        "pass_threshold": getattr(settings, "CAPTCHA_PASS_THRESHOLD", 0.7),
        "challenge_threshold": getattr(settings, "CAPTCHA_CHALLENGE_THRESHOLD", 0.3),
    }


@router.put("/captcha/config", tags=["admin-captcha"])
async def update_captcha_config(
    body: dict,
    current_user: User = Depends(require_user),
):
    """캡챠 가중치/임계값 런타임 변경

    body 예시:
      {"lstm_weight": 0.7, "knn_weight": 0.2,
       "pass_threshold": 0.7, "challenge_threshold": 0.3}
    """
    import services.captcha_service as cs

    updated: list[str] = []

    # ── 가중치 변경 ──
    if "lstm_weight" in body or "knn_weight" in body:
        lstm_w = float(body.get("lstm_weight", settings.LSTM_WEIGHT))
        knn_w = float(body.get("knn_weight", getattr(settings, "KNN_WEIGHT", 0.2)))
        rule_w = 1.0 - lstm_w - knn_w

        if rule_w < 0 or rule_w > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"rule_weight({rule_w:.2f})가 0~1 범위를 벗어납니다. lstm+knn <= 1.0 이어야 합니다.",
            )

        settings.LSTM_WEIGHT = lstm_w
        settings.KNN_WEIGHT = knn_w
        updated.append(f"weights: rule={rule_w:.0%} KNN={knn_w:.0%} LSTM={lstm_w:.0%}")

    # ── 임계값 변경 ──
    if "pass_threshold" in body:
        val = float(body["pass_threshold"])
        settings.CAPTCHA_PASS_THRESHOLD = val
        cs.CAPTCHA_PASS_THRESHOLD = val
        updated.append(f"pass_threshold={val}")

    if "challenge_threshold" in body:
        val = float(body["challenge_threshold"])
        settings.CAPTCHA_CHALLENGE_THRESHOLD = val
        cs.CAPTCHA_CHALLENGE_THRESHOLD = val
        updated.append(f"challenge_threshold={val}")

    lstm_w = settings.LSTM_WEIGHT
    knn_w = getattr(settings, "KNN_WEIGHT", 0.2)
    rule_w = round(1.0 - lstm_w - knn_w, 4)

    return {
        "message": f"변경 완료: {', '.join(updated)}" if updated else "변경 사항 없음",
        "lstm_weight": lstm_w,
        "knn_weight": knn_w,
        "rule_weight": rule_w,
        "pass_threshold": settings.CAPTCHA_PASS_THRESHOLD,
        "challenge_threshold": settings.CAPTCHA_CHALLENGE_THRESHOLD,
    }


# ── 챌린지 강제 발동 ──────────────────────────────────────

@router.post("/captcha/force-challenge", tags=["admin-captcha"])
async def force_challenge(
    body: dict | None = None,
    current_user: User = Depends(require_user),
):
    """특정 IP(또는 모든 IP)에 대해 다음 캡챠를 강제로 challenge로 만듦.

    body 예시:
      {"ip": "202.31.255.26"}   → 해당 IP만
      {} 또는 body 없음          → 모든 활성 IP
    """
    target_ip = (body or {}).get("ip", "").strip()

    if target_ip:
        # 특정 IP만
        await redis_client.setex(f"captcha:force-challenge:{target_ip}", 300, "1")
        return {"message": f"{target_ip}에 챌린지 강제 발동 (5분간 유효)"}

    # IP 미지정 시: 와일드카드 대신 잘 알려진 방법 → 0.0.0.0 플래그
    await redis_client.setex("captcha:force-challenge:*", 300, "1")
    return {"message": "모든 IP에 챌린지 강제 발동 (5분간 유효)"}

# ─── 결제 내역 관리 ───────────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel

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
    commissionRate: float
    commissionAmount: int
    paymentMethod: str | None
    status: str
    billingMonth: str
    pricingType: str | None
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


@router.get("/payments", response_model=AdminPaymentListOut)
async def get_admin_payments(
    keyword: str = Query(""),
    status_filter: str = Query("", alias="status"),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """결제 내역 전체 목록 (관리자 전용, 페이지네이션)"""
    stmt = (
        select(Payment, User, Party, Service)
        .join(User, Payment.user_id == User.id)
        .join(Party, Payment.party_id == Party.id)
        .outerjoin(Service, Party.service_id == Service.id)
        .order_by(Payment.created_at.desc())
    )

    if status_filter:
        stmt = stmt.where(func.lower(Payment.status) == status_filter.lower())
    if date_from:
        stmt = stmt.where(func.date(Payment.created_at) >= date_from)
    if date_to:
        stmt = stmt.where(func.date(Payment.created_at) <= date_to)

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
                basePrice=payment.base_price or payment.amount,
                amount=payment.amount,
                discountReason=payment.discount_reason,
                commissionRate=float(payment.commission_rate or 0.10),
                commissionAmount=payment.commission_amount,
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
