from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from core.database import get_db
from core.security import get_current_user
from schemas.mypage.profile import (
    MyPageProfileResponse,
    UpdateMyPageProfileResponse,
)
from services.mypage.profile_service import (
    get_my_profile_service,
    update_my_profile_service,
)

router = APIRouter(tags=["mypage-profile"])

@router.get("/users/me/profile", response_model=MyPageProfileResponse)
async def get_my_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await get_my_profile_service(
        db=db,
        current_user=current_user,
    )

@router.patch("/users/me/profile", response_model=UpdateMyPageProfileResponse)
async def update_my_profile(
    nickname: str = Form(...),
    phone: str = Form(...),
    profile_image: Optional[UploadFile] = File(default=None),
    remove_profile_image: bool = Form(default=False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await update_my_profile_service(
        db=db,
        current_user=current_user,
        nickname=nickname,
        phone=phone,
        profile_image=profile_image,
        remove_profile_image=remove_profile_image,
    )
