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