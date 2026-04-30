from fastapi import HTTPException, status
from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from uuid import UUID

from models.user import User, UserReferrer
from models.admin import ActivityLog


async def update_admin_user_recommender_service(
    db: AsyncSession,
    *,
    target_user_id: UUID,
    referrer_nickname: str | None,
    admin_user_id: UUID,
    reason: str | None = None,
):
    target_result = await db.execute(
        select(User).where(User.id == target_user_id)
    )
    target_user = target_result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="대상 사용자를 찾을 수 없습니다.",
        )

    # 추천인 제거
    if not referrer_nickname or not referrer_nickname.strip():
        await db.execute(
            delete(UserReferrer).where(UserReferrer.user_id == target_user_id)
        )

        target_user.referrer_id = None
        target_user.referrer_count = 0

        db.add(
            ActivityLog(
                actor_user_id=admin_user_id,
                action_type="ADMIN_USER_REFERRER_UPDATE",
                description=f"관리자가 사용자({target_user.nickname})의 추천인을 제거했습니다.",
                target_id=target_user_id,
                extra_metadata={
                    "reason": reason,
                    "referrerNickname": None,
                },
            )
        )

        await db.commit()
        await db.refresh(target_user)
        return target_user

    nickname = referrer_nickname.strip()

    referrer_result = await db.execute(
        select(User).where(
            User.nickname == nickname,
            User.is_active.is_(True),
        )
    )
    referrer_user = referrer_result.scalar_one_or_none()

    if not referrer_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="존재하지 않는 추천인이거나 비활성화된 사용자입니다.",
        )

    if referrer_user.id == target_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="자기 자신은 추천인으로 설정할 수 없습니다.",
        )

    # 관리자 변경은 실무상 '교체'가 편함: 기존 추천인 전체 삭제 후 1명으로 재설정
    await db.execute(
        delete(UserReferrer).where(UserReferrer.user_id == target_user_id)
    )

    db.add(
        UserReferrer(
            user_id=target_user_id,
            referrer_id=referrer_user.id,
        )
    )

    target_user.referrer_id = referrer_user.id
    target_user.referrer_count = 1

    db.add(
        ActivityLog(
            actor_user_id=admin_user_id,
            action_type="ADMIN_USER_REFERRER_UPDATE",
            description=f"관리자가 사용자({target_user.nickname})의 추천인을 {referrer_user.nickname}(으)로 변경했습니다.",
            target_id=target_user_id,
            extra_metadata={
                "reason": reason,
                "referrerId": str(referrer_user.id),
                "referrerNickname": referrer_user.nickname,
            },
        )
    )

    await db.commit()
    await db.refresh(target_user)

    return target_user



# 운영/제재 이력
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.admin import ActivityLog, ModerationAction
from models.report import Report
from models.user import User


def _meta(row: ActivityLog) -> dict[str, Any]:
    return row.extra_metadata or {}


def _operation_type_from_activity(action_type: str) -> str | None:
    if action_type.startswith("STATUS_"):
        return "STATUS_CHANGE"
    if action_type == "TRUST_SCORE_UPDATED":
        return "TRUST_SCORE_CHANGE"
    if action_type in {"REFERRER_UPDATED", "ADMIN_USER_REFERRER_UPDATE"}:
        return "RECOMMENDER_CHANGE"
    return None


async def get_admin_user_operation_logs_service(
    db: AsyncSession,
    *,
    target_user_id: UUID,
    limit: int = 50,
) -> dict:
    target_user = await db.get(User, target_user_id)
    if not target_user:
        return {"logs": [], "total": 0}

    activity_rows = (
        await db.execute(
            select(ActivityLog)
            .where(ActivityLog.target_id == target_user_id)
            .where(
                ActivityLog.action_type.in_(
                    [
                        "STATUS_정상",
                        "STATUS_주의",
                        "STATUS_정지",
                        "TRUST_SCORE_UPDATED",
                        "REFERRER_UPDATED",
                        "ADMIN_USER_REFERRER_UPDATE",
                    ]
                )
            )
            .order_by(ActivityLog.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()

    moderation_rows = (
        await db.execute(
            select(ModerationAction)
            .where(ModerationAction.user_id == target_user_id)
            .order_by(ModerationAction.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()

    report_rows = (
        await db.execute(
            select(Report)
            .where(
                (Report.target_id == target_user_id)
                | (Report.reporter_id == target_user_id)
            )
            .order_by(Report.created_at.desc())
            .limit(limit)
        )
    ).scalars().all()

    logs: list[dict] = []

    for row in activity_rows:
        log_type = _operation_type_from_activity(row.action_type)
        if not log_type:
            continue

        metadata = _meta(row)

        logs.append(
            {
                "id": str(row.id),
                "type": log_type,
                "userId": str(target_user_id),
                "beforeStatus": metadata.get("beforeStatus"),
                "afterStatus": (
                    row.action_type.replace("STATUS_", "")
                    if row.action_type.startswith("STATUS_")
                    else metadata.get("afterStatus")
                ),
                "beforeTrustScore": metadata.get("beforeTrustScore"),
                "afterTrustScore": metadata.get("afterTrustScore"),
                "beforeRecommenderId": metadata.get("beforeRecommenderId"),
                "afterRecommenderId": metadata.get("afterRecommenderId"),
                "reason": metadata.get("reason") or row.description,
                "adminId": str(row.actor_user_id) if row.actor_user_id else None,
                "createdAt": row.created_at.isoformat() if row.created_at else "",
            }
        )

    for row in moderation_rows:
        logs.append(
            {
                "id": str(row.id),
                "type": "SANCTION",
                "userId": str(target_user_id),
                "sanctionType": row.action_type,
                "sanctionDurationDays": (
                    row.duration_minutes // 1440
                    if row.duration_minutes
                    else None
                ),
                "reason": row.reason,
                "adminId": str(row.admin_id) if row.admin_id else None,
                "createdAt": row.created_at.isoformat() if row.created_at else "",
            }
        )

    for row in report_rows:
        is_reporter = row.reporter_id == target_user_id

        logs.append(
            {
                "id": str(row.id),
                "type": "REPORT_CREATED" if is_reporter else "REPORT_RECEIVED",
                "userId": str(target_user_id),
                "reportId": str(row.id),
                "reportReason": row.category,
                "reportTargetUserId": (
                    str(row.target_id)
                    if row.target_type == "USER"
                    else None
                ),
                "reason": row.description,
                "adminId": str(row.reviewed_by) if row.reviewed_by else None,
                "createdAt": row.created_at.isoformat() if row.created_at else "",
            }
        )

    logs.sort(key=lambda item: item["createdAt"], reverse=True)

    return {
        "logs": logs[:limit],
        "total": len(logs),
    }