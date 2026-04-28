from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel


class ReferrerOut(BaseModel):
    id: str
    nickname: str

    model_config = {"from_attributes": True}


class RecentActivityItem(BaseModel):
    id: str
    action: str
    description: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: dict[str, Any] = {}
    target_id: Optional[str] = None
    created_at: Optional[datetime] = None


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

    # 나를 추천인으로 등록한 사용자 수
    recommendation_count: int = 0

    # 내가 등록한 추천인 목록
    referrers: list[ReferrerOut] = []
    referrer_count: int = 0

    recent_activities: list[RecentActivityItem] = []


class UpdateMyPageProfileResponse(BaseModel):
    message: str
    user: MyPageProfileResponse


class DeleteMyAccountRequest(BaseModel):
    password: Optional[str] = None


class DeleteMyAccountResponse(BaseModel):
    message: str