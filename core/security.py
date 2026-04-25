import uuid
from typing import Optional
from jose import JWTError, jwt
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
    """
    토큰이 없거나 유효하지 않아도 에러를 던지지 않고 None을 반환합니다.
    비로그인 유저도 파티 목록 조회 등에 접근 가능하도록 합니다.
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
    except (JWTError, ValueError):
        return None


async def get_current_user(
    current_user: Optional[User] = Depends(get_current_user_optional),
) -> Optional[User]:
    return current_user


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
