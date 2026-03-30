import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Cookie, HTTPException, Response, status, Request
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.refresh_token import RefreshToken
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


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = utc_now() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update(
        {
            "exp": expire,
            "type": "access",
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = utc_now() + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update(
        {
            "exp": expire,
            "type": "refresh",
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="올바른 access token이 아닙니다.",
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 access token입니다.",
        )


def decode_refresh_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="올바른 refresh token이 아닙니다.",
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 refresh token입니다.",
        )


def set_access_token_cookie(response: Response, access_token: str) -> None:
    response.set_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def clear_access_token_cookie(response: Response) -> None:
    response.delete_cookie(
        key=ACCESS_TOKEN_COOKIE_NAME,
        path="/",
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
    )


def set_refresh_token_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/",
    )


def clear_refresh_token_cookie(response: Response) -> None:
    response.delete_cookie(
        key=REFRESH_TOKEN_COOKIE_NAME,
        path="/",
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
    )


async def create_and_store_refresh_token(
    db: AsyncSession,
    user: User,
    request: Request,
    family_id: Optional[uuid.UUID] = None,
    parent_token_id: Optional[uuid.UUID] = None,
) -> str:
    refresh_token_id = uuid.uuid4()
    refresh_family_id = family_id or uuid.uuid4()
    expires_at = utc_now() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    refresh_token = create_refresh_token(
        data={
            "sub": str(user.id),
            "jti": str(refresh_token_id),
            "family_id": str(refresh_family_id),
        }
    )

    token_row = RefreshToken(
        id=refresh_token_id,
        user_id=user.id,
        token_hash=hash_token(refresh_token),
        family_id=refresh_family_id,
        parent_token_id=parent_token_id,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
        expires_at=expires_at,
        revoked_at=None,
        revoke_reason=None,
        created_at=utc_now(),
    )
    db.add(token_row)
    await db.flush()

    return refresh_token


async def issue_tokens_and_save(
    response: Response,
    db: AsyncSession,
    user: User,
    request: Request,
) -> None:
    access_token = create_access_token(data={"sub": str(user.id)})
    refresh_token = await create_and_store_refresh_token(
        db=db,
        user=user,
        request=request,
    )

<<<<<<< Updated upstream
    user.last_login_at = utc_now()
=======
    user.last_login_at = datetime.utcnow()  # ← 수정
>>>>>>> Stashed changes
    await db.commit()

    set_access_token_cookie(response, access_token)
    set_refresh_token_cookie(response, refresh_token)


async def rotate_refresh_token(
    db: AsyncSession,
    response: Response,
    request: Request,
    raw_refresh_token: str,
) -> None:
    payload = decode_refresh_token(raw_refresh_token)

    user_id_str = payload.get("sub")
    jti = payload.get("jti")
    family_id_str = payload.get("family_id")

    if not user_id_str or not jti or not family_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token payload가 올바르지 않습니다.",
        )

    try:
        user_id = uuid.UUID(user_id_str)
        token_id = uuid.UUID(jti)
        family_id = uuid.UUID(family_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token 식별값 형식이 올바르지 않습니다.",
        )

    result = await db.execute(
        select(RefreshToken).where(RefreshToken.id == token_id)
    )
    token_row = result.scalar_one_or_none()

    if not token_row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="저장되지 않은 refresh token입니다.",
        )

    if token_row.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token 사용자 정보가 일치하지 않습니다.",
        )

    if token_row.token_hash != hash_token(raw_refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token 검증에 실패했습니다.",
        )

    if token_row.revoked_at is not None:
        await db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.family_id == token_row.family_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(
                revoked_at=utc_now(),
                revoke_reason="refresh token reuse detected",
            )
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이미 폐기된 refresh token입니다.",
        )

    if token_row.expires_at <= utc_now():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="만료된 refresh token입니다.",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없거나 비활성화 상태입니다.",
        )

    token_row.revoked_at = utc_now()
    token_row.revoke_reason = "rotated"

    new_refresh_token = await create_and_store_refresh_token(
        db=db,
        user=user,
        request=request,
        family_id=family_id,
        parent_token_id=token_row.id,
    )

    new_access_token = create_access_token(data={"sub": str(user.id)})
    user.last_login_at = utc_now()

    await db.commit()

    set_access_token_cookie(response, new_access_token)
    set_refresh_token_cookie(response, new_refresh_token)


async def revoke_refresh_token(
    db: AsyncSession,
    raw_refresh_token: str,
    reason: str = "logout",
) -> None:
    try:
        payload = decode_refresh_token(raw_refresh_token)
    except HTTPException:
        return

    jti = payload.get("jti")
    if not jti:
        return

    try:
        token_id = uuid.UUID(jti)
    except ValueError:
        return

    result = await db.execute(
        select(RefreshToken).where(RefreshToken.id == token_id)
    )
    token_row = result.scalar_one_or_none()

    if not token_row:
        return

    if token_row.revoked_at is None:
        token_row.revoked_at = utc_now()
        token_row.revoke_reason = reason
        await db.commit()


def get_current_user_id(
    access_token: Optional[str] = Cookie(default=None, alias=ACCESS_TOKEN_COOKIE_NAME),
) -> str:
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다.",
        )

    payload = decode_access_token(access_token)
    user_id = payload.get("sub")

    if not user_id:
<<<<<<< Updated upstream
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="토큰에 사용자 정보가 없습니다.",
        )

=======
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="토큰에 사용자 정보가 없습니다.")
>>>>>>> Stashed changes
    return user_id