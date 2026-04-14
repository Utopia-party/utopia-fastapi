import uuid
from pydantic import BaseModel, Field
from typing import Optional, List


class CategoryOut(BaseModel):
    id: uuid.UUID
    name: str

    model_config = {"from_attributes": True}


class ServiceOut(BaseModel):
    """파티 생성 시 서비스 선택에 사용"""
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
    # DB: max_members NOT NULL → service.max_members에서 채워줌 (선택 가능)
    max_members: Optional[int] = Field(None, ge=2, le=10)
    # DB: monthly_per_person NOT NULL → service.monthly_price에서 채워줌
    monthly_per_person: Optional[int] = Field(None, ge=0)
    min_trust_score: Optional[float] = Field(0.0, ge=0)
    start_date: Optional[str] = None   # "YYYY-MM-DD"
    end_date: Optional[str] = None     # "YYYY-MM-DD"


class PartyOut(BaseModel):
    id: uuid.UUID
    leader_id: Optional[uuid.UUID]
    service_id: Optional[uuid.UUID]
    title: str
    status: Optional[str]
    host_nickname: Optional[str] = None
    service_name: Optional[str] = None
    category_name: Optional[str] = None
    max_members: Optional[int] = None
    monthly_price: Optional[int] = None
    logo_image_key: Optional[str] = None
    logo_image_url: Optional[str] = None
    member_count: int = 0
    is_joined: bool = False

    model_config = {"from_attributes": True}


class PartyListOut(BaseModel):
    parties: List[PartyOut]
    total: int
    page: int
    size: int
