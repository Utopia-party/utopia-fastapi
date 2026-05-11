import uuid
from typing import Optional
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from core.config import settings
from core.database import get_db
from models.user import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

async def get_current_user_optional(
    access_token: Optional[str] = Cookie(default=None, alias="access_token"),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    if not access_token:
        return None
    try:
        payload = jwt.decode(access_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "access":
            return None

        user_id_str: str = payload.get("sub", "")
        user_id = uuid.UUID(user_id_str)

        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    except (JWTError, ValueError):
        return None


async def get_current_user(
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Optional[User]:
    return current_user


async def get_current_user_or_raise_expired(
    access_token: Optional[str] = Cookie(default=None, alias="access_token"),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    토큰이 없으면 None 반환 (비로그인 허용).
    토큰이 있지만 만료된 경우엔 401을 던져 프론트가 refresh 후 재시도하게 합니다.
    이의제기처럼 로그인 필수지만 정지 유저도 접근 가능한 엔드포인트에 사용합니다.
    """
    if not access_token:
        return None
    try:
        payload = jwt.decode(access_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "access":
            return None

        user_id_str: str = payload.get("sub", "")
        user_id = uuid.UUID(user_id_str)

        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    except ExpiredSignatureError:
        # 만료된 토큰 → 프론트에서 refresh 후 재시도할 수 있도록 401
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="만료된 access token입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (JWTError, ValueError):
        return None


async def require_user(
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> User:
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다."
        )

    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="비활성화된 계정입니다."
        )
    return current_user
