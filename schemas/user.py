import uuid
from pydantic import BaseModel
from typing import Optional


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    nickname: str

    model_config = {"from_attributes": True}


# 마이페이지 - 프로필
class MyPageProfileResponse(BaseModel):
    email: str
    nickname: str
    # ✅ Fix: User 모델에서 phone은 Optional[str] → None일 수 있으므로 Optional로 변경
    phone: Optional[str] = None

    model_config = {"from_attributes": True}
