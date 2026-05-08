from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.notification import Notification
from schemas.notification import NotificationOut
from services.notification_ws_service import notification_connection_manager
from services.notifications.email_notification_service import enqueue_email_notification


def serialize_notification(notification: Notification) -> NotificationOut:
    return NotificationOut.model_validate(notification)


async def get_unread_notification_count_service(
    db: AsyncSession,
    user_id: UUID,
) -> int:
    result = await db.execute(
        select(func.count(Notification.id)).where(
            Notification.user_id == user_id,
            Notification.is_read.is_(False),
        )
    )
    return int(result.scalar_one() or 0)


async def get_my_notifications_service(
    db: AsyncSession,
    user_id: UUID,
    limit: int = 20,
) -> list[Notification]:
    result = await db.execute(
        select(Notification)
        .where(Notification.user_id == user_id)
        .order_by(Notification.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_latest_notifications_service(
    db: AsyncSession,
    user_id: UUID,
    limit: int = 10,
) -> list[Notification]:
    result = await db.execute(
        select(Notification)
        .where(Notification.user_id == user_id)
        .order_by(Notification.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def create_notification_service(
    db: AsyncSession,
    *,
    user_id: UUID,
    type: str,
    title: str,
    message: str,
    reference_type: str | None = None,
    reference_id: UUID | None = None,
    metadata: dict | None = None,
    commit: bool = True,
) -> Notification:
    notification = Notification(
        user_id=user_id,
        type=type,
        title=title,
        message=message,
        reference_type=reference_type,
        reference_id=reference_id,
        meta=metadata,
        is_read=False,
        read_at=None,
    )

    db.add(notification)

    if commit:
        await db.commit()
        await db.refresh(notification)

    return notification


async def push_notification_created_event(
    db: AsyncSession,
    notification: Notification,
) -> None:
    unread_count = await get_unread_notification_count_service(
        db=db,
        user_id=notification.user_id,
    )

    await notification_connection_manager.send_to_user(
        notification.user_id,
        {
            "type": "notification_created",
            "notification": serialize_notification(notification).model_dump(
                mode="json",
                by_alias=True,
            ),
            "unread_count": unread_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def push_notification_updated_event(
    db: AsyncSession,
    notification: Notification,
) -> None:
    unread_count = await get_unread_notification_count_service(
        db=db,
        user_id=notification.user_id,
    )

    await notification_connection_manager.send_to_user(
        notification.user_id,
        {
            "type": "notification_updated",
            "notification": serialize_notification(notification).model_dump(
                mode="json",
                by_alias=True,
            ),
            "unread_count": unread_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def push_notification_deleted_event(
    db: AsyncSession,
    *,
    user_id: UUID,
    notification_id: UUID,
) -> None:
    unread_count = await get_unread_notification_count_service(
        db=db,
        user_id=user_id,
    )

    await notification_connection_manager.send_to_user(
        user_id,
        {
            "type": "notification_deleted",
            "notification_id": str(notification_id),
            "unread_count": unread_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def notify_user(
    db: AsyncSession,
    *,
    user_id: UUID,
    type: str,
    title: str,
    message: str,
    reference_type: str | None = None,
    reference_id: UUID | None = None,
    metadata: dict | None = None,
) -> Notification:
    """
    1. 알림 DB 저장
    2. 실시간 웹소켓 발행
    3. 이메일 대상 이벤트면 Celery 이메일 발송
    """
    notification = await create_notification_service(
        db=db,
        user_id=user_id,
        type=type,
        title=title,
        message=message,
        reference_type=reference_type,
        reference_id=reference_id,
        metadata=metadata,
        commit=True,
    )

    await push_notification_created_event(
        db=db,
        notification=notification,
    )

    event_code = None
    if metadata:
        event_code = metadata.get("event_code")

    await enqueue_email_notification(
        db=db,
        user_id=user_id,
        event_code=event_code,
        title=title,
        message=message,
        metadata=metadata,
    )

    return notification


async def mark_notification_as_read_service(
    db: AsyncSession,
    user_id: UUID,
    notification_id: UUID,
) -> Notification:
    result = await db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == user_id,
        )
    )
    notification = result.scalar_one_or_none()

    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="해당 알림을 찾을 수 없습니다.",
        )

    if not notification.is_read:
        notification.is_read = True
        notification.read_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(notification)

    unread_count = await get_unread_notification_count_service(
        db=db,
        user_id=user_id,
    )

    await notification_connection_manager.send_to_user(
        user_id,
        {
            "type": "notification_read",
            "notification": serialize_notification(notification).model_dump(
                mode="json",
                by_alias=True,
            ),
            "notification_id": str(notification.id),
            "unread_count": unread_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return notification


async def mark_all_notifications_as_read_service(
    db: AsyncSession,
    user_id: UUID,
) -> int:
    now = datetime.now(timezone.utc)

    count_result = await db.execute(
        select(func.count(Notification.id)).where(
            Notification.user_id == user_id,
            Notification.is_read.is_(False),
        )
    )
    unread_count = int(count_result.scalar_one() or 0)

    if unread_count == 0:
        notifications = await get_my_notifications_service(
            db=db,
            user_id=user_id,
            limit=20,
        )

        await notification_connection_manager.send_to_user(
            user_id,
            {
                "type": "notifications_read_all",
                "notifications": [
                    serialize_notification(item).model_dump(
                        mode="json",
                        by_alias=True,
                    )
                    for item in notifications
                ],
                "unread_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 0

    await db.execute(
        update(Notification)
        .where(
            Notification.user_id == user_id,
            Notification.is_read.is_(False),
        )
        .values(
            is_read=True,
            read_at=now,
        )
    )
    await db.commit()

    notifications = await get_my_notifications_service(
        db=db,
        user_id=user_id,
        limit=20,
    )

    await notification_connection_manager.send_to_user(
        user_id,
        {
            "type": "notifications_read_all",
            "notifications": [
                serialize_notification(item).model_dump(
                    mode="json",
                    by_alias=True,
                )
                for item in notifications
            ],
            "unread_count": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    return unread_count