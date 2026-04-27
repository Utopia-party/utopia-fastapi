import uuid, re
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


def normalize_phone(v: str) -> str:
    return re.sub(r"[^0-9]", "", v)


# 일반 회원가입
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    name: str = Field(..., min_length=1, max_length=50)
    nickname: str = Field(..., min_length=2, max_length=50)
    phone: str = Field(..., min_length=10, max_length=13)
    referrer: Optional[str] = None

    # 비밀번호 유효성검사
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str):
        regex = r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[!@#$%^&*()_+{}\[\]:;<>,.?~\\/-]).{8,}$"
        if not re.match(regex, v):
            raise ValueError("비밀번호는 8자 이상, 영문/숫자/특수문자를 포함해야 합니다.")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str):
        normalized = normalize_phone(v)
        if len(normalized) not in (10, 11):
            raise ValueError("휴대폰 번호는 10~11자리 숫자여야 합니다.")
        return normalized

# 일반 로그인
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: Optional[str] = None
    nickname: str
    phone: Optional[str] = None
    referrer: Optional[str] = None
    model_config = {"from_attributes": True}

# 소셜 로그인
class SocialLoginBody(BaseModel):
    oauth: str = Field(..., min_length=1, max_length=20)
    code: str = Field(..., min_length=1)
    state: Optional[str] = None

    @field_validator("oauth")
    @classmethod
    def validate_oauth(cls, v: str):
        oauth = v.lower().strip()
        if oauth not in {"google", "kakao", "naver"}:
            raise ValueError("지원하지 않는 소셜 로그인입니다.")
        return oauth


class SocialSignupBody(BaseModel):
    oauth: str = Field(..., min_length=1, max_length=20)
    oauth_id: str = Field(..., min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    name: Optional[str] = Field(default=None, max_length=50)
    nickname: str = Field(..., min_length=2, max_length=50)
    phone: Optional[str] = Field(min_length=10, max_length=13)

    @field_validator("oauth")
    @classmethod
    def validate_oauth(cls, v: str):
        oauth = v.lower().strip()
        if oauth not in {"google", "kakao", "naver"}:
            raise ValueError("지원하지 않는 소셜 로그인입니다.")
        return oauth

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]):
        if v is None or v == "":
            return None

        normalized = normalize_phone(v)
        if len(normalized) not in (10, 11):
            raise ValueError("휴대폰 번호는 10~11자리 숫자여야 합니다.")
        return normalized



# 이메일 찾기
class FindIdRequest(BaseModel):
    nickname: str = Field(..., min_length=1, max_length=50)
    phone: str = Field(..., min_length=10, max_length=13)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str):
        normalized = normalize_phone(v)
        if len(normalized) not in (10, 11):
            raise ValueError("휴대폰 번호는 10~11자리 숫자여야 합니다.")
        return normalized


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
