import uuid
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional


class UserCreate(BaseModel):
    email: EmailStr
    nickname: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6)
    phone: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    nickname: str
    role: str
    trust_score: float
    is_active: bool
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ✅ Fix: UserResponse 중복 정의 제거 → schemas/user.py의 UserResponse 단일 사용
#         auth.py에 있던 UserResponse 클래스 삭제
