import uuid, re
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

# 회원가입
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    name: str = Field(..., min_length=1, max_length=50)
    nickname: str = Field(..., min_length=2, max_length=50)
    phone: str = Field(..., min_length=10, max_length=11)
    referrer: Optional[str] = None

    # 비밀번호 유효성검사
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str):
        regex = r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*()_+{}\[\]:;<>,.?~\\/-]).{8,}$"
        if not re.match(regex, v):
            raise ValueError("비밀번호는 8자 이상, 영문/숫자/특수문자를 포함해야 합니다.")
        return v

class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: Optional[str] = None
    nickname: str
    phone: Optional[str] = None
    referrer: Optional[str] = None
    model_config = {"from_attributes": True}

# 이메일 찾기
class FindIdRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    phone: str = Field(..., min_length=10, max_length=11)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str):
        if not v.isdigit():
            raise ValueError("휴대폰 번호는 숫자만 입력해주세요.")
        return v

class FindIdResponse(BaseModel):
    email: Optional[EmailStr] = None
    message: Optional[str] = None

# 비밀번호 찾기
class FindPasswordRequest(BaseModel):
    email: EmailStr

class FindPasswordResponse(BaseModel):
    message: str

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str):
        regex = r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*()_+{}\[\]:;<>,.?~\\/-]).{8,}$"
        if not re.match(regex, v):
            raise ValueError("비밀번호는 8자 이상, 영문/숫자/특수문자를 포함해야 합니다.")
        return v

class ResetPasswordResponse(BaseModel):
    message: str    

# 일반 로그인
class UserLogin(BaseModel):
    email: EmailStr
    password: str

