import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# 파티 채팅
class MessageOut(BaseModel):
    message: str

    model_config = {"from_attributes": True}


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    name: Optional[str] = None
    nickname: str
    role: str
    trust_score: float
    is_active: bool
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class MyPageProfileResponse(BaseModel):
    email: str
    name: Optional[str] = None
    nickname: str
    phone: Optional[str] = None
    referrer: Optional[str] = None

    model_config = {"from_attributes": True}


from typing import List
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class ReferrerOut(BaseModel):
    id: UUID
    nickname: str

    model_config = {"from_attributes": True}


class MyReferrersResponse(BaseModel):
    referrers: List[ReferrerOut]
    referrer_count: int


class UpdateMyReferrersRequest(BaseModel):
    referrers: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("referrers")
    @classmethod
    def validate_referrers(cls, v: list[str]):
        cleaned = [item.strip() for item in v if item and item.strip()]

        if len(cleaned) > 5:
            raise ValueError("추천인은 최대 5명까지 등록할 수 있습니다.")

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("중복된 추천인이 있습니다.")

        return cleaned


class UpdateMyReferrersResponse(BaseModel):
    message: str
    referrers: List[ReferrerOut]