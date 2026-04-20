from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.mypage.trust_history import MyTrustHistoryResponse
from services.mypage.trust_history_service import get_my_trust_history_service

router = APIRouter(tags=["mypage-trust-history"])


@router.get("/users/me/trust-history", response_model=MyTrustHistoryResponse)
async def get_my_trust_history(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await get_my_trust_history_service(
        db=db,
        current_user=current_user,
    )