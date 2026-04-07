import random
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, Response, Cookie, Request
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.config import settings
from core.database import get_db
from core.redis_client import redis_client
from models.user import User
from models.refresh_token import RefreshToken
from schemas import UserCreate, UserOut, UserResponse, UserLogin
from services.auth_service import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
    rotate_refresh_token,
    handle_refresh_token_reuse,
    set_access_token_cookie,
    set_refresh_token_cookie,
    clear_access_token_cookie,
    clear_refresh_token_cookie,
    issue_tokens_and_save,
    hash_refresh_token,
)
from services.oauth_service import (
    get_google_access_token, get_google_user_info,
    get_kakao_access_token, get_kakao_user_info,
    get_naver_access_token, get_naver_user_info,
)

router = APIRouter(tags=["auth"])

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


# ✅ Fix: social_login / social_signup body를 raw dict 대신 Pydantic 스키마로 교체
class SocialLoginBody(BaseModel):
    oauth: str
    code: str
    state: Optional[str] = None


class SocialSignupBody(BaseModel):
    oauth: str
    oauth_id: str
    email: Optional[str] = None
    nickname: str


def get_email_auth_key(email: str) -> str:
    return f"email_auth:{email}"


# ─── Refresh token ─────────────────────────────────────────────
@router.post("/refresh")
async def refresh_token_api(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="refresh token이 없습니다.",
        )

    token_hash = hash_refresh_token(refresh_token)

    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    )
    token_row = result.scalar_one_or_none()

    if not token_row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 refresh token입니다.",
        )

    if token_row.revoked_at is not None:
        await handle_refresh_token_reuse(db, token_row)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="재사용된 refresh token이 감지되어 모든 세션이 종료되었습니다.",
        )

    if token_row.expires_at <= datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="만료된 refresh token입니다.",
        )

    new_access_token = create_access_token(data={"sub": str(token_row.user_id)})

    new_refresh_token = await rotate_refresh_token(
        db=db,
        old_token_row=token_row,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    set_access_token_cookie(response, new_access_token)
    set_refresh_token_cookie(response, new_refresh_token)

    return {"message": "access token과 refresh token이 재발급되었습니다."}


# ─── 로그인 상태 확인 ────────────────────────────────────────
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
        "user": {
            "user_id": str(user.id),
            "email": user.email,
            "nickname": user.nickname,
            "provider": user.provider,
            "role": user.role,
        },
    }


# ─── 이메일 인증번호 요청 ────────────────────────────────────
@router.post("/email-request")
async def email_request(email: EmailStr, background_tasks: BackgroundTasks):
    auth_code = str(random.randint(100000, 999999))
    # ✅ Fix: 동기 redis → await 추가
    await redis_client.setex(get_email_auth_key(str(email)), settings.EMAIL_AUTH_TTL_SECONDS, auth_code)

    message = MessageSchema(
        subject="[Party-Up] 회원가입 인증번호입니다.",
        recipients=[email],
        body=f"안녕하세요! Party-Up 서비스 가입을 위한 인증번호는 [{auth_code}] 입니다.",
        subtype="plain",
    )
    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, message)
    return {"message": "인증 메일이 발송되었습니다.", "expires_in": settings.EMAIL_AUTH_TTL_SECONDS}


# ─── 이메일 인증번호 확인 ────────────────────────────────────
@router.post("/email-verify")
async def email_verify(email: str, code: str):
    redis_key = get_email_auth_key(email)
    # ✅ Fix: 동기 redis → await 추가
    saved_code = await redis_client.get(redis_key)
    if not saved_code:
        raise HTTPException(status_code=400, detail="인증번호가 없거나 만료되었습니다.")
    if saved_code != code:
        raise HTTPException(status_code=400, detail="인증번호가 틀렸습니다.")
    await redis_client.delete(redis_key)
    return {"success": True, "message": "이메일 인증에 성공했습니다."}


# ─── 이메일 중복검사 ─────────────────────────────────────────
@router.get("/users/check-email")
async def check_email(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == email))
    return {"exists": result.scalar_one_or_none() is not None}


# ─── 닉네임 중복검사 ────────────────────────────────────────
@router.get("/users/check-nickname")
async def check_nickname(nickname: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.nickname == nickname))
    return {"exists": result.scalar_one_or_none() is not None}


# ─── 회원가입 ───────────────────────────────────────────────
@router.post("/users", response_model=UserResponse, status_code=201)
async def signup(
    # 상원: 회원가입 직후 자동 로그인 쿠키를 발급하려고 Request 객체를 함께 받습니다.
    request: Request,  # 상원
    # 상원: issue_tokens_and_save가 쿠키를 심을 수 있도록 Response 객체를 함께 받습니다.
    response: Response,  # 상원
    # 상원: 프론트가 보낸 회원가입 바디를 UserCreate 스키마로 검증해 받습니다.
    user: UserCreate,  # 상원
    # 상원: 중복 검사, 사용자 생성, 토큰 저장에 쓸 DB 세션을 주입받습니다.
    db: AsyncSession = Depends(get_db),  # 상원
):
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
    # 상원: 회원가입 직후 바로 관심사 저장 API를 사용할 수 있도록 자동 로그인 쿠키를 발급합니다.
    await issue_tokens_and_save(  # 상원
        # 상원: 발급된 액세스/리프레시 토큰 쿠키를 심을 대상 Response입니다.
        response=response,  # 상원
        # 상원: refresh token 저장과 사용자 조회에 사용할 DB 세션입니다.
        db=db,  # 상원
        # 상원: 방금 생성한 사용자를 토큰 발급 대상 계정으로 넘깁니다.
        user=new_user,  # 상원
        # 상원: 세션 기록에 남길 user-agent를 현재 요청 헤더에서 읽어 넘깁니다.
        user_agent=request.headers.get("user-agent"),  # 상원
        # 상원: 세션 기록에 남길 클라이언트 IP를 현재 요청에서 읽어 넘깁니다.
        ip_address=request.client.host if request.client else None,  # 상원
    )  # 상원
    return new_user


# ─── 일반 로그인 ────────────────────────────────────────────
@router.post("/login")
async def login(
    request: Request,
    user_credentials: UserLogin,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
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

    await issue_tokens_and_save(
        response=response,
        db=db,
        user=user,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    return {
        "message": "로그인에 성공했습니다.",
        "user": {
            "email": user.email,
            "nickname": user.nickname,
            "role": user.role,
        },
    }


# ─── 로그아웃 ───────────────────────────────────────────────
@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    refresh_token = request.cookies.get("refresh_token")

    if refresh_token:
        token_hash = hash_refresh_token(refresh_token)
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        token_row = result.scalar_one_or_none()

        if token_row and token_row.revoked_at is None:
            token_row.revoked_at = datetime.now(timezone.utc)
            token_row.revoke_reason = "logout"
            await db.commit()

    clear_access_token_cookie(response)
    clear_refresh_token_cookie(response)
    return {"message": "로그아웃 되었습니다."}


# ─── OAuth 유저정보 조회 헬퍼 ──────────────────────────────
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


# ─── OAuth 로그인 ───────────────────────────────────────────
# ✅ Fix: data: dict → SocialLoginBody Pydantic 스키마로 교체
@router.post("/auth/login")
async def social_login(data: SocialLoginBody, response: Response, db: AsyncSession = Depends(get_db)):
    oauth = data.oauth.lower().strip()
    code = data.code
    state = data.state

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


# ─── OAuth 회원가입 ─────────────────────────────────────────
# ✅ Fix: data: dict → SocialSignupBody Pydantic 스키마로 교체
@router.post("/auth/social/signup")
async def social_signup(data: SocialSignupBody, response: Response, db: AsyncSession = Depends(get_db)):
    oauth = data.oauth.lower().strip()
    oauth_id = data.oauth_id
    email = data.email
    nickname = data.nickname.strip()

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
