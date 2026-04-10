from sqlalchemy import select
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import require_user
from models.notification import Notification
from models.user import User
from schemas import NotificationOut

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/latest", response_model=list[NotificationOut])
async def get_latest_notifications():
    # Server DB does not support broadcast notices in `notifications`
    # because `user_id` is required. Keep the endpoint stable for the
    # frontend banner and return an empty list until a dedicated notice
    # store is introduced.
    return []


@router.get("/me", response_model=list[NotificationOut])
async def get_my_notifications(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Notification)
        # ✅ Fix: user_id → current_user.id (UUID)
        .where(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
        .limit(20)
    )
    return result.scalars().all()
