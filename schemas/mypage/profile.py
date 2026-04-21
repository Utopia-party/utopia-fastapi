from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RecentActivityItem(BaseModel):
    id: str
    action: str
    description: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    target_id: Optional[str] = None
    created_at: datetime


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

    total_party_participations: int = 0
    active_party_count: int = 0
    recommendation_count: int = 0
    recent_activities: list[RecentActivityItem] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class UpdateMyPageProfileResponse(BaseModel):
    message: str = Field(default="프로필이 수정되었습니다.")
    user: MyPageProfileResponse