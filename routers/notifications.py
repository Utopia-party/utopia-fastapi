from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from core.database import get_db
from core.security import require_user
from models.notification import Notification
from models.user import User
from schemas.schemas import NotificationOut

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/latest", response_model=list[NotificationOut])
async def get_latest_notifications(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Notification)
        .where(Notification.user_id == None)  # noqa - 시스템 공지
        .order_by(Notification.created_at.desc())
        .limit(5)
    )
    return result.scalars().all()


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
