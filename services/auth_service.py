import secrets   # 토큰 랜덤문자열 생성용
import hashlib   # 토큰 문자열-> 해시값 변경
import uuid

from sqlalchemy import select
from sqlalchemy import update
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Cookie, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from models.refresh_token import RefreshToken
from models.user import User
from core.config import settings

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS

COOKIE_SECURE = settings.COOKIE_SECURE
COOKIE_SAMESITE = settings.COOKIE_SAMESITE

ACCESS_TOKEN_COOKIE_NAME = "access_token"
REFRESH_TOKEN_COOKIE_NAME = "refresh_token"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 비밀번호 해시
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# 로그인 시 비밀번호 검증
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# --------------------------------------------------------------
# access token관련
# --------------------------------------------------------------

# ----------------------access token 생성 ----------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ----------------------access token 검증 ----------------------
def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="올바른 access token이 아닙니다.")
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유효하지 않거나 만료된 access token입니다.")

# ----------------------access token 브라우저 저장 ----------------------
def set_access_token_cookie(response: Response, access_token: str) -> None:
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME, value=access_token,
        httponly=True, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, path="/",
    )

# ----------------------access token 브라우저 삭제 ----------------------
def clear_access_token_cookie(response: Response) -> None:
    response.delete_cookie(key=ACCESS_TOKEN_COOKIE_NAME, path="/", samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)


# --------------------------------------------------------------
# refresh token관련
# --------------------------------------------------------------

# ----------------------refresh token 생성 ----------------------
# refresh token 랜덤문자열 생성
def create_refresh_token() -> str:
    return secrets.token_urlsafe(32)

# refresh token 해시
def hash_refresh_token(token: str) -> str:  
    return hashlib.sha256(token.encode()).hexdigest()

# refresh token 만료시간 계산
def get_refresh_token_expiry(expires_delta: Optional[timedelta] = None) -> datetime:
    return datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )

# ----------------------refresh token DB 저장 ----------------------
async def issue_tokens_and_save(
    response: Response,
    db: AsyncSession,
    user: User,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> None:
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = create_refresh_token()

    refresh_expires_at = get_refresh_token_expiry()
    refresh_token_hash = hash_refresh_token(refresh_token)
    family_id = uuid.uuid4()

    refresh_token_row = RefreshToken(
        user_id=user.id,
        token_hash=refresh_token_hash,
        family_id=family_id,
        parent_token_id=None,
        user_agent=user_agent,
        ip_address=ip_address,
        expires_at=refresh_expires_at,
        revoked_at=None,
        revoke_reason=None,
        created_at=datetime.now(timezone.utc),
    )

    db.add(refresh_token_row)

    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()

    # 쿠키에 access/refresh token 넣음
    set_access_token_cookie(response, access_token)
    set_refresh_token_cookie(response, refresh_token)

# ----------------------refresh token 폐기 ----------------------
def revoke_refresh_token(token_row: RefreshToken, reason: str) -> None:
    token_row.revoked_at = datetime.now(timezone.utc)
    token_row.revoke_reason = reason

# ----------------------refresh token 브라우저 저장 ----------------------
def set_refresh_token_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME, value=refresh_token,
        httponly=True, secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60, path="/",
    )

# ----------------------refresh token 브라우저 삭제 ----------------------
def clear_refresh_token_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_TOKEN_COOKIE_NAME, path="/", samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)


# ----------------------refresh token 재발급 ----------------------
async def rotate_refresh_token(
    db: AsyncSession,
    old_token_row: RefreshToken,
    user_agent: str | None = None,
    ip_address: str | None = None,
) -> str:
    revoke_refresh_token(old_token_row, "rotated")

    new_refresh_token = create_refresh_token()
    new_refresh_token_hash = hash_refresh_token(new_refresh_token)

    new_token_row = RefreshToken(
        user_id=old_token_row.user_id,
        token_hash=new_refresh_token_hash,
        family_id=old_token_row.family_id,
        parent_token_id=old_token_row.id,
        user_agent=user_agent,
        ip_address=ip_address,
        expires_at=get_refresh_token_expiry(),
        revoked_at=None,
        revoke_reason=None,
        created_at=datetime.now(timezone.utc),
    )

    db.add(new_token_row)
    await db.commit()

    return new_refresh_token    

# ----------------------refresh token 재발급 세션 폐기 ----------------------
async def revoke_token_family(
    db: AsyncSession,
    family_id: uuid.UUID,
    reason: str,
) -> None:
    await db.execute(
        update(RefreshToken)
        .where(
            RefreshToken.family_id == family_id,
            RefreshToken.revoked_at.is_(None),
        )
        .values(
            revoked_at=datetime.now(timezone.utc),
            revoke_reason=reason,
        )
    )
    await db.commit()

# ----------------------refresh token 재사용 감지 헬퍼 ----------------------
async def handle_refresh_token_reuse(
    db: AsyncSession,
    token_row: RefreshToken,
) -> None:
    await revoke_token_family(
        db=db,
        family_id=token_row.family_id,
        reason="token_reuse_detected",
    )


# def get_current_user_email(
#     access_token: Optional[str] = Cookie(default=None, alias=ACCESS_TOKEN_COOKIE_NAME),
# ) -> str:
#     if not access_token:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다.")
#     payload = decode_access_token(access_token)
#     user_id = payload.get("sub")
#     if not user_id:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="토큰에 사용자 정보가 없습니다.")
#     return user_id