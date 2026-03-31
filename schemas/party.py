import uuid
from pydantic import BaseModel, Field
from typing import Optional


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

    model_config = {"from_attributes": True}


class PartyListOut(BaseModel):
    parties: list[PartyOut]
    total: int
    page: int
    size: int