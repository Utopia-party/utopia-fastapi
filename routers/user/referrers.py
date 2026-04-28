from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user
from models.user import User
from schemas.user import (
    MyReferrersResponse,
    ReferrerOut,
    UpdateMyReferrersRequest,
    UpdateMyReferrersResponse,
)
from services.auth_service import (
    get_my_referrers_service,
    replace_user_referrers_service,
)

router = APIRouter(prefix="/users/me/referrers", tags=["referrers"])


@router.get("", response_model=MyReferrersResponse)
async def get_my_referrers(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    referrers = await get_my_referrers_service(
        db=db,
        user_id=current_user.id,
    )

    return MyReferrersResponse(
        referrers=[
            ReferrerOut(
                id=user.id,
                nickname=user.nickname,
            )
            for user in referrers
        ],
        referrer_count=len(referrers),
    )


@router.patch("", response_model=UpdateMyReferrersResponse)
async def update_my_referrers(
    payload: UpdateMyReferrersRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    referrers = await replace_user_referrers_service(
        db=db,
        user_id=current_user.id,
        referrer_nicknames=payload.referrers,
    )

    return UpdateMyReferrersResponse(
        message="추천인 정보가 변경되었습니다.",
        referrers=[
            ReferrerOut(
                id=user.id,
                nickname=user.nickname,
            )
            for user in referrers
        ],
    )