import uuid
from pydantic import BaseModel, Field
from typing import Optional, List

# ✅ 추가된 카테고리 응답 스키마
class CategoryOut(BaseModel):
    id: int
    name: str

    model_config = {"from_attributes": True}

class PartyCreate(BaseModel):
    service_id: uuid.UUID
    title: str = Field(..., min_length=2, max_length=200)

class PartyOut(BaseModel):
    id: uuid.UUID
    leader_id: Optional[uuid.UUID]
    service_id: Optional[uuid.UUID]
    title: str
    status: Optional[str]          # ← PartyStatusEnum 대신 str로 변경
    host_nickname: Optional[str] = None
    service_name: Optional[str] = None
    category_name: Optional[str] = None
    max_members: Optional[int] = None
    monthly_price: Optional[int] = None
    logo_image_key: Optional[str] = None
    member_count: int = 0
    
    # ✅ 추가된 필드: 현재 로그인한 사용자가 해당 파티원인지 여부
    is_joined: bool = False 

    model_config = {"from_attributes": True}

class PartyListOut(BaseModel):
    parties: List[PartyOut]
    total: int
    page: int
    size: int
