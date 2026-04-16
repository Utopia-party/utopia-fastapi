import uuid
from typing import Optional, List

from pydantic import BaseModel, Field


class CategoryOut(BaseModel):
    name: str
    model_config = {"from_attributes": True}


class ServiceOut(BaseModel):
    id: uuid.UUID
    name: str
    category: str
    max_members: int
    monthly_price: int
    logo_image_url: Optional[str] = None
    model_config = {"from_attributes": True}


class PartyCreate(BaseModel):
    service_id: uuid.UUID
    title: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    max_members: Optional[int] = Field(None, ge=2, le=10)
    min_trust_score: Optional[float] = Field(0.0, ge=0)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    captcha_pass_token: str = Field(..., min_length=1)


class PartyOut(BaseModel):
    id: uuid.UUID
    leader_id: Optional[uuid.UUID]
    service_id: Optional[uuid.UUID]
    title: str
    status: Optional[str]
    host_nickname: Optional[str] = None
    host_trust_score: Optional[float] = None
    service_name: Optional[str] = None
    category_name: Optional[str] = None
    max_members: Optional[int] = None
    monthly_price: Optional[int] = None
    original_price: Optional[int] = None
    service_total_price: Optional[int] = None  # 서비스 전체 구독 금액 (환급 계산용)
    logo_image_key: Optional[str] = None
    logo_image_url: Optional[str] = None
    member_count: int = 0
    is_joined: bool = False
    # 현재 로그인 유저의 해당 파티 참여 상태
    # None | 'pending' | 'active' | 'kicked' | 'left' | 'rejected' | 'leader'
    my_member_status: Optional[str] = None
    model_config = {"from_attributes": True}


class PartyListOut(BaseModel):
    parties: List[PartyOut]
    total: int
    page: int
    size: int


# ---- 내 파티 / 멤버 관리용 (v2 추가) ----

class MyPartyOut(PartyOut):
    """내 파티 목록 전용 — is_owner 플래그 포함."""
    is_owner: bool = False


class MyPartyListOut(BaseModel):
    parties: List[MyPartyOut]


class PartyMemberOut(BaseModel):
    user_id: uuid.UUID
    nickname: Optional[str] = None
    role: str  # 'leader' | 'member'
    is_current_user: bool = False

    model_config = {"from_attributes": True}


class PartyMembersOut(BaseModel):
    members: List[PartyMemberOut]


class TransferLeaderRequest(BaseModel):
    new_leader_user_id: uuid.UUID
