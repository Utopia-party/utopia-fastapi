from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# 마이페이지 프로필 수정 (프로필이미지 / 닉네임 / 휴대전화 수정 가능)
class MyPageProfileResponse(BaseModel):
    user_id: str
    email: str
    name: Optional[str] = None
    nickname: str
    phone: Optional[str] = None
    provider: str
    role: str
    trust_score: float
    profile_image: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UpdateMyPageProfileResponse(BaseModel):
    message: str = Field(default="프로필이 수정되었습니다.")
    user: MyPageProfileResponse