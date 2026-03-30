import random
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Response, Cookie, Request
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.config import settings
from core.database import get_db
from core.redis_client import redis_client
from models.user import User
from schemas.schemas import UserCreate, UserOut, UserLogin, UserResponse
from services.auth_service import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    set_access_token_cookie,
    set_refresh_token_cookie,
    clear_access_token_cookie,
    clear_refresh_token_cookie,
    issue_tokens_and_save,
)
from services.oauth_service import (
    get_google_access_token, get_google_user_info,
    get_kakao_access_token, get_kakao_user_info,
    get_naver_access_token, get_naver_user_info,
)

router = APIRouter(tags=["auth"])

# ✅ Fix: os.getenv → settings로 통일
conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
)


def get_email_auth_key(email: str) -> str:
    return f"email_auth:{email}"


# ─── 토큰 재발급 ──────────────────────────────────────────────────
@router.post("/refresh")
async def refresh_token_api(request: Request, response: Response):
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="refresh token이 없습니다.")

    payload = decode_refresh_token(refresh_token)
    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="refresh token에 사용자 정보가 없습니다.")

    new_access_token = create_access_token(data={"sub": user_id_str})
    set_access_token_cookie(response, new_access_token)
    return {"message": "access token이 재발급되었습니다."}


# ─── 로그인 상태 확인 ─────────────────────────────────────────────
@router.get("/me")
async def me(
    access_token: str | None = Cookie(default=None, alias="access_token"),
    db: AsyncSession = Depends(get_db),
):
    if not access_token:
        return {"is_logged_in": False, "user": None}
    try:
        payload = decode_access_token(access_token)
        user_id_str = payload.get("sub")
        if not user_id_str:
            return {"is_logged_in": False, "user": None}
        user_id = uuid.UUID(user_id_str)
    except Exception:
        return {"is_logged_in": False, "user": None}

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return {"is_logged_in": False, "user": None}

    return {
        "is_logged_in": True,
        "user": {"user_id": str(user.id), "email": user.email, "nickname": user.nickname, "provider": user.provider},
    }


# ─── 이메일 인증번호 요청 ─────────────────────────────────────────
@router.post("/email-request")
async def email_request(email: EmailStr, background_tasks: BackgroundTasks):
    auth_code = str(random.randint(100000, 999999))
    # ✅ Fix: settings.EMAIL_AUTH_TTL_SECONDS 사용
    redis_client.setex(get_email_auth_key(str(email)), settings.EMAIL_AUTH_TTL_SECONDS, auth_code)

    message = MessageSchema(
        subject="[Party-Up] 회원가입 인증번호입니다.",
        recipients=[email],
        body=f"안녕하세요! Party-Up 서비스 가입을 위한 인증번호는 [{auth_code}] 입니다.",
        subtype="plain",
    )
    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, message)
    return {"message": "인증 메일이 발송되었습니다.", "expires_in": settings.EMAIL_AUTH_TTL_SECONDS}


# ─── 이메일 인증번호 확인 ─────────────────────────────────────────
@router.post("/email-verify")
async def email_verify(email: str, code: str):
    redis_key = get_email_auth_key(email)
    saved_code = redis_client.get(redis_key)
    if not saved_code:
        raise HTTPException(status_code=400, detail="인증번호가 없거나 만료되었습니다.")
    if saved_code != code:
        raise HTTPException(status_code=400, detail="인증번호가 틀렸습니다.")
    redis_client.delete(redis_key)
    return {"success": True, "message": "이메일 인증에 성공했습니다."}


# ─── 이메일 중복검사 ──────────────────────────────────────────────
@router.get("/users/check-email")
async def check_email(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == email))
    return {"exists": result.scalar_one_or_none() is not None}


# ─── 닉네임 중복검사 ─────────────────────────────────────────────
@router.get("/users/check-nickname")
async def check_nickname(nickname: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.nickname == nickname))
    return {"exists": result.scalar_one_or_none() is not None}


# ─── 회원가입 ────────────────────────────────────────────────────
@router.post("/users", response_model=UserResponse, status_code=201)
async def signup(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 등록된 이메일입니다.")

    result = await db.execute(select(User).where(User.nickname == user.nickname))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 사용 중인 닉네임입니다.")

    new_user = User(
        email=user.email,
        nickname=user.nickname,
        password_hash=get_password_hash(user.password),
        phone=user.phone,
        provider="local",
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user


# ─── 일반 로그인 ─────────────────────────────────────────────────
@router.post("/login")
async def login(user_credentials: UserLogin, response: Response, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user_credentials.email))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다.")
    if user.provider != "local" or user.password_hash is None:
        raise HTTPException(status_code=400, detail="소셜 로그인으로 가입한 계정입니다.")
    if not verify_password(user_credentials.password, user.password_hash):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="비활성화된 계정입니다.")

    await issue_tokens_and_save(response, db, user)
    return {"message": "로그인에 성공했습니다.", "user": {"email": user.email, "nickname": user.nickname}}


# ─── 로그아웃 ────────────────────────────────────────────────────
@router.post("/logout")
async def logout(response: Response):
    clear_access_token_cookie(response)
    clear_refresh_token_cookie(response)
    return {"message": "로그아웃 되었습니다."}


# ─── OAuth 유저정보 조회 헬퍼 ────────────────────────────────────
def get_oauth_user_info(oauth: str, code: str, state: Optional[str] = None):
    oauth = oauth.lower().strip()
    if oauth == "google":
        token = get_google_access_token(code)
        info = get_google_user_info(token)
        return str(info.get("sub")), info.get("email")
    elif oauth == "kakao":
        token = get_kakao_access_token(code)
        info = get_kakao_user_info(token)
        account = info.get("kakao_account", {}) or {}
        return str(info.get("id")), account.get("email")
    elif oauth == "naver":
        if not state:
            raise HTTPException(status_code=400, detail="네이버 로그인에는 state 값이 필요합니다.")
        token = get_naver_access_token(code, state)
        info = get_naver_user_info(token)
        return str(info.get("id")), info.get("email")
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 소셜 로그인입니다.")


# ─── OAuth 로그인 ────────────────────────────────────────────────
@router.post("/auth/login")
async def social_login(data: dict, response: Response, db: AsyncSession = Depends(get_db)):
    oauth = data.get("oauth")
    code = data.get("code")
    state = data.get("state")

    if not oauth or not code:
        raise HTTPException(status_code=400, detail="oauth와 code는 필수입니다.")

    oauth = str(oauth).lower().strip()
    oauth_id, email = get_oauth_user_info(oauth, code, state)

    result = await db.execute(select(User).where(User.provider == oauth, User.provider_id == oauth_id))
    user = result.scalar_one_or_none()
    if user:
        if not user.is_active:
            raise HTTPException(status_code=403, detail="비활성화된 계정입니다.")
        await issue_tokens_and_save(response, db, user)
        return {"status": "LOGIN_SUCCESS", "message": "소셜 로그인에 성공했습니다.", "user": {"email": user.email, "nickname": user.nickname}}

    if email:
        result = await db.execute(select(User).where(User.email == email))
        email_user = result.scalar_one_or_none()
        if email_user:
            if email_user.provider not in ("local", oauth) and email_user.provider_id:
                raise HTTPException(status_code=400, detail="이미 다른 소셜 계정에 연결된 이메일입니다.")
            email_user.provider = oauth
            email_user.provider_id = oauth_id
            await db.commit()
            await db.refresh(email_user)
            await issue_tokens_and_save(response, db, email_user)
            return {"status": "LOGIN_SUCCESS", "message": "기존 계정과 소셜 로그인이 연동되었습니다.", "user": {"email": email_user.email, "nickname": email_user.nickname}}

    return {"status": "NEED_NICKNAME", "oauth": oauth, "oauth_id": oauth_id, "email": email}


# ─── OAuth 회원가입 ───────────────────────────────────────────────
@router.post("/auth/social/signup")
async def social_signup(data: dict, response: Response, db: AsyncSession = Depends(get_db)):
    oauth = data.get("oauth")
    oauth_id = data.get("oauth_id")
    email = data.get("email")
    nickname = data.get("nickname")

    if not oauth or not oauth_id or not nickname:
        raise HTTPException(status_code=400, detail="oauth, oauth_id, nickname은 필수입니다.")

    oauth = str(oauth).lower().strip()
    nickname = nickname.strip()

    result = await db.execute(select(User).where(User.provider == oauth, User.provider_id == oauth_id))
    existing = result.scalar_one_or_none()
    if existing:
        await issue_tokens_and_save(response, db, existing)
        return {"status": "LOGIN_SUCCESS", "message": "이미 가입된 소셜 계정입니다.", "user": {"email": existing.email, "nickname": existing.nickname}}

    result = await db.execute(select(User).where(User.nickname == nickname))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 사용 중인 닉네임입니다.")

    if not email:
        email = f"{oauth}_{oauth_id}@social.local"

    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 같은 이메일로 가입된 계정이 있습니다.")

    user = User(email=email, nickname=nickname, provider=oauth, provider_id=oauth_id, password_hash=None)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    await issue_tokens_and_save(response, db, user)

    return {"status": "SIGNUP_SUCCESS", "message": "소셜 회원가입 및 로그인에 성공했습니다.", "user": {"email": user.email, "nickname": user.nickname}}
