from typing import Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile,Request, Response, Body
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from core.database import get_db
from core.security import get_current_user
from schemas.mypage.profile import (
    MyPageProfileResponse,
    UpdateMyPageProfileResponse,
    DeleteMyAccountRequest,
    DeleteMyAccountResponse
)
from services.mypage.profile_service import (
    get_my_profile_service,
    update_my_profile_service,
    delete_my_account_service
)
from services.auth_service import (
    clear_access_token_cookie,
    clear_refresh_token_cookie,
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


# 회원 탈퇴
@router.delete("/users/me", response_model=DeleteMyAccountResponse)
async def delete_my_account(
    request: Request,
    response: Response,
    payload: DeleteMyAccountRequest | None = Body(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await delete_my_account_service(
        db=db,
        current_user=current_user,
        password=payload.password if payload else None,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    clear_access_token_cookie(response)
    clear_refresh_token_cookie(response)

    return result