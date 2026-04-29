from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
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



@dataclass
class AdminContext:
    user: User
    role: AdminRole


def _format_datetime(value: datetime | None) -> str:
    if not value:
        return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M")


def _format_relative(value: datetime | None) -> str:
    if not value:
        return "-"

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
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
    return datetime(value.year, value.month, value.day)


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
            "can_view_dashboard": True,
            "can_manage_users": True,
            "can_manage_services": True,
            "can_manage_parties": True,
            "can_manage_quick_match": True,
            "can_manage_reports": True,
            "can_manage_moderation": True,
            "can_manage_captcha": True,
            "can_approve_settlements": True,
            "can_manage_payments": True,
            "can_manage_handocr": True,
            "can_view_logs": True,
            "can_view_cloud_monitoring": True,
            "can_manage_admins": True,
        }
    return {
        "can_view_dashboard": True,
        "can_manage_users": True,
        "can_manage_services": True,
        "can_manage_parties": True,
        "can_manage_quick_match": True,
        "can_manage_reports": True,
        "can_manage_moderation": True,
        "can_manage_captcha": True,
        "can_approve_settlements": True,
        "can_manage_payments": True,
        "can_manage_handocr": True,
        "can_view_logs": True,
        "can_view_cloud_monitoring": True,
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
        "can_view_dashboard": payload.canViewDashboard,
        "can_manage_users": payload.canManageUsers,
        "can_manage_services": payload.canManageServices,
        "can_manage_parties": payload.canManageParties,
        "can_manage_quick_match": payload.canManageQuickMatch,
        "can_manage_reports": payload.canManageReports,
        "can_manage_moderation": payload.canManageChatModeration,
        "can_manage_captcha": payload.canManageCaptcha,
        "can_approve_settlements": payload.canApproveSettlements,
        "can_manage_payments": payload.canManagePayments,
        "can_manage_handocr": payload.canManageHandOcr,
        "can_view_logs": payload.canViewLogs,
        "can_view_cloud_monitoring": payload.canViewCloudMonitoring,
        "can_manage_admins": payload.canManageAdmins,
    }


def _has_any_admin_permission(values: dict[str, bool]) -> bool:
    return any(values.values())


def _serialize_admin_permissions(role: AdminRole) -> AdminPermissionOut:
    return AdminPermissionOut(
        canViewDashboard=role.can_view_dashboard,
        canManageUsers=role.can_manage_users,
        canManageServices=role.can_manage_services,
        canManageParties=role.can_manage_parties,
        canManageQuickMatch=role.can_manage_quick_match,
        canManageReports=role.can_manage_reports,
        canManageChatModeration=role.can_manage_moderation,
        canManageCaptcha=role.can_manage_captcha,
        canApproveSettlements=role.can_approve_settlements,
        canManagePayments=role.can_manage_payments,
        canManageHandOcr=role.can_manage_handocr,
        canViewLogs=role.can_view_logs,
        canViewCloudMonitoring=role.can_view_cloud_monitoring,
        canManageAdmins=role.can_manage_admins,
    )


def _serialize_admin_role(role: AdminRole, user: User, created_by: User | None) -> AdminRoleRecordOut:
    return AdminRoleRecordOut(
        id=str(role.id),
        userId=str(user.id),
        adminId=user.nickname or user.email,
        canViewDashboard=role.can_view_dashboard,
        canManageUsers=role.can_manage_users,
        canManageServices=role.can_manage_services,
        canManageParties=role.can_manage_parties,
        canManageQuickMatch=role.can_manage_quick_match,
        canManageReports=role.can_manage_reports,
        canManageChatModeration=role.can_manage_moderation,
        canManageCaptcha=role.can_manage_captcha,
        canApproveSettlements=role.can_approve_settlements,
        canManagePayments=role.can_manage_payments,
        canManageHandOcr=role.can_manage_handocr,
        canViewLogs=role.can_view_logs,
        canViewCloudMonitoring=role.can_view_cloud_monitoring,
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
        commissionRate=(
            float(service.commission_rate)
            if service.commission_rate is not None
            else 0.30
        ),
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


async def require_admin_dashboard_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_view_dashboard", "대시보드 조회 권한이 없습니다.")


async def require_admin_service_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_services", "구독 서비스 관리 권한이 없습니다.")


async def require_admin_party_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_parties", "파티 관리 권한이 없습니다.")


async def require_admin_quick_match_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_quick_match", "빠른매칭 관리 권한이 없습니다.")


async def require_admin_report_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_reports", "신고 관리 권한이 없습니다.")


async def require_admin_receipt_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_captcha", "캡챠 관리 권한이 없습니다.")


async def require_admin_settlement_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_approve_settlements", "정산 승인 권한이 없습니다.")


async def require_admin_payment_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_payments", "수익내역 관리 권한이 없습니다.")


async def require_admin_handocr_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_handocr", "HandOCR CAPTCHA 관리 권한이 없습니다.")


async def require_admin_log_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_view_logs", "시스템 로그 조회 권한이 없습니다.")


async def require_admin_cloud_monitor_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(
        admin,
        "can_view_cloud_monitoring",
        "클라우드 모니터링 조회 권한이 없습니다.",
    )


async def require_admin_moderation_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_moderation", "모더레이션 관리 권한이 없습니다.")


async def require_admin_role_permission(
    admin: AdminContext = Depends(require_admin_context),
) -> AdminContext:
    return _assert_admin_permission(admin, "can_manage_admins", "관리자 권한 변경 권한이 없습니다.")
