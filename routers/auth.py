import random
import uuid
from datetime import datetime, timezone
from typing import Optional, Any

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
from models.mypage.trust_score import TrustScore
from models.admin import ActivityLog
from schemas.auth import (
    UserCreate,
    UserResponse,
    UserLogin,
    FindIdRequest,
    FindIdResponse,
    FindPasswordRequest,
    FindPasswordResponse,
    ResetPasswordRequest,
    ResetPasswordResponse,
    SocialLoginBody,
    SocialSignupBody,
)
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
from services.mypage.profile_service import (
    _build_profile_image_url,
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

def get_email_auth_key(email: str) -> str:
    return f"email_auth:{email}"


def build_initial_trust_score_history(user_id, trust_score_value: float) -> TrustScore:
    return TrustScore(
        user_id=user_id,
        previous_score=0.0,
        new_score=trust_score_value,
        change_amount=trust_score_value,
        reason="가입 초기 신뢰도 부여",
        reference_id=None,
        created_by=user_id,
    )


def _safe_metadata(metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return metadata or {}


async def create_activity_log(
    db: AsyncSession,
    user_id: uuid.UUID,
    action: str,
    description: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    target_id: Optional[uuid.UUID] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    log = ActivityLog(
        actor_user_id=user_id,
        action_type=action,
        description=description,
        extra_metadata=_safe_metadata(metadata),
        target_id=target_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(log)
    await db.commit()


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

    # timezone-aware 비교 (DB에서 naive datetime으로 올 경우 UTC로 처리)
    expires_at = token_row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="만료된 refresh token입니다.",
        )

    client_ip = request.client.host if request.client else None
    if client_ip:
        is_ip_banned = await redis_client.get(f"ip:banned:{client_ip}")
        if is_ip_banned:
            raise HTTPException(status_code=403, detail="이용이 제한된 계정입니다.")

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

    if not user.is_active:
        return {"is_logged_in": False, "user": None}

    return {
        "is_logged_in": True,
        "user": {
            "user_id": str(user.id),
            "email": user.email,
            "name": user.name,
            "nickname": user.nickname,
            "provider": user.provider,
            "role": user.role,
            "phone": user.phone,
            "trust_score": user.trust_score,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "profile_image": _build_profile_image_url(user.profile_image_key),
        },
    }


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

            await create_activity_log(
                db=db,
                user_id=token_row.user_id,
                action="LOGOUT",
                description="로그아웃",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

    clear_access_token_cookie(response)
    clear_refresh_token_cookie(response)
    return {"message": "로그아웃 되었습니다."}


async def get_oauth_user_info(oauth: str, code: str, state: Optional[str] = None):
    oauth = oauth.lower().strip()
    if oauth == "google":
        token =  get_google_access_token(code)
        info = await get_google_user_info(token)
        return str(info.get("sub")), info.get("email"), info.get("name")
    elif oauth == "kakao":
        token = await get_kakao_access_token(code)
        info = await get_kakao_user_info(token)
        account = info.get("kakao_account", {}) or {}
        profile = account.get("profile", {}) or {}
        return str(info.get("id")), account.get("email"), profile.get("nickname")
    elif oauth == "naver":
        if not state:
            raise HTTPException(status_code=400, detail="네이버 로그인에는 state 값이 필요합니다.")
        token = get_naver_access_token(code, state)
        info = await get_naver_user_info(token)
        return str(info.get("id")), info.get("email"), info.get("name") or info.get("nickname")
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 소셜 로그인입니다.")


@router.post("/auth/login")
async def social_login(data: SocialLoginBody, response: Response, request: Request, db: AsyncSession = Depends(get_db)):
    oauth = data.oauth.lower().strip()
    code = data.code
    state = data.state

    oauth_id, email, name = await get_oauth_user_info(oauth, code, state)

    result = await db.execute(select(User).where(User.provider == oauth, User.provider_id == oauth_id))
    user = result.scalar_one_or_none()
    if user:
        client_ip = request.client.host if request.client else None
        if client_ip:
            is_ip_banned = await redis_client.get(f"ip:banned:{client_ip}")
            if is_ip_banned:
                raise HTTPException(status_code=403, detail="이용이 제한된 계정입니다.")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="비활성화된 계정입니다.")

        await issue_tokens_and_save(
            response=response,
            db=db,
            user=user,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None,
        )

        await create_activity_log(
            db=db,
            user_id=user.id,
            action="LOGIN",
            description="로그인",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return {
            "status": "LOGIN_SUCCESS",
            "message": "소셜 로그인에 성공했습니다.",
            "user": {"email": user.email, "nickname": user.nickname},
        }

    if email:
        result = await db.execute(select(User).where(User.email == email))
        email_user = result.scalar_one_or_none()
        if email_user:
            if email_user.provider not in ("local", oauth) and email_user.provider_id:
                raise HTTPException(status_code=400, detail="이미 다른 소셜 계정에 연결된 이메일입니다.")

            email_user.provider = oauth
            email_user.provider_id = oauth_id
            if not (email_user.name or "").strip():
                email_user.name = (name or email_user.nickname or "").strip() or None
            await db.commit()
            await db.refresh(email_user)

            await issue_tokens_and_save(
                response=response,
                db=db,
                user=email_user,
                user_agent=request.headers.get("user-agent"),
                ip_address=request.client.host if request.client else None,
            )

            await create_activity_log(
                db=db,
                user_id=email_user.id,
                action="LOGIN",
                description="로그인",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

            return {
                "status": "LOGIN_SUCCESS",
                "message": "기존 계정과 소셜 로그인이 연동되었습니다.",
                "user": {"email": email_user.email, "nickname": email_user.nickname},
            }

    return {
        "status": "NEED_NICKNAME",
        "oauth": oauth,
        "oauth_id": oauth_id,
        "email": email,
        "name": name,
    }


@router.post("/auth/social/signup")
async def social_signup(data: SocialSignupBody, response: Response, request: Request, db: AsyncSession = Depends(get_db)):
    client_ip = request.client.host if request.client else None
    if client_ip:
        is_ip_banned = await redis_client.get(f"ip:banned:{client_ip}")
        if is_ip_banned:
            raise HTTPException(status_code=403, detail="이용이 제한된 계정입니다.")
    oauth = data.oauth.lower().strip()
    oauth_id = data.oauth_id
    email = data.email
    name = (data.name or "").strip()
    nickname = data.nickname.strip()
    phone = data.phone

    result = await db.execute(select(User).where(User.provider == oauth, User.provider_id == oauth_id))
    existing = result.scalar_one_or_none()
    if existing:
        await issue_tokens_and_save(
            response=response,
            db=db,
            user=existing,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None,
        )

        await create_activity_log(
            db=db,
            user_id=existing.id,
            action="LOGIN",
            description="로그인",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return {
            "status": "LOGIN_SUCCESS",
            "message": "이미 가입된 소셜 계정입니다.",
            "user": {"email": existing.email, "nickname": existing.nickname},
        }

    result = await db.execute(select(User).where(User.nickname == nickname))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 사용 중인 닉네임입니다.")

    if not email:
        email = f"{oauth}_{oauth_id}@social.local"

    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 같은 이메일로 가입된 계정이 있습니다.")

    user = User(
        email=email,
        name=name,
        nickname=nickname,
        provider=oauth,
        provider_id=oauth_id,
        password_hash=None,
        phone=phone,
    )
    db.add(user)
    await db.flush()

    initial_trust_history = build_initial_trust_score_history(
        user_id=user.id,
        trust_score_value=float(user.trust_score),
    )
    db.add(initial_trust_history)

    await db.commit()
    await db.refresh(user)

    await issue_tokens_and_save(
        response=response,
        db=db,
        user=user,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    await create_activity_log(
        db=db,
        user_id=user.id,
        action="LOGIN",
        description="로그인",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    return {
        "status": "SIGNUP_SUCCESS",
        "message": "소셜 회원가입 및 로그인에 성공했습니다.",
        "user": {"email": user.email, "nickname": user.nickname},
    }


@router.post("/users", response_model=UserResponse, status_code=201)
async def signup(
    request: Request,
    response: Response,
    user: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    client_ip = request.client.host if request.client else None
    if client_ip:
        is_ip_banned = await redis_client.get(f"ip:banned:{client_ip}")
        if is_ip_banned:
            raise HTTPException(status_code=403, detail="이용이 제한된 계정입니다.")
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 등록된 이메일입니다.")

    result = await db.execute(select(User).where(User.nickname == user.nickname))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 사용 중인 닉네임입니다.")

    referrer_id = None

    if user.referrer:
        result = await db.execute(
            select(User).where(User.nickname == user.referrer)
        )
        ref_user = result.scalar_one_or_none()

        if not ref_user:
            raise HTTPException(status_code=400, detail="존재하지 않는 추천인입니다.")

        referrer_id = ref_user.id

    new_user = User(
        email=user.email,
        name=user.name,
        nickname=user.nickname,
        password_hash=get_password_hash(user.password),
        phone=user.phone,
        referrer_id=referrer_id,
        provider="local",
    )
    db.add(new_user)
    await db.flush()

    initial_trust_history = build_initial_trust_score_history(
        user_id=new_user.id,
        trust_score_value=float(new_user.trust_score),
    )
    db.add(initial_trust_history)

    await db.commit()
    await db.refresh(new_user)

    await issue_tokens_and_save(
        response=response,
        db=db,
        user=new_user,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    await create_activity_log(
        db=db,
        user_id=new_user.id,
        action="LOGIN",
        description="로그인",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    return new_user


@router.get("/users/check-email")
async def check_email(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == email))
    return {"exists": result.scalar_one_or_none() is not None}


@router.post("/email-request")
async def email_request(
    email: EmailStr,
    background_tasks: BackgroundTasks,
    type: str = "signup",
    db: AsyncSession = Depends(get_db),
):
    if type == "reset-password":
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()

        if not user or user.provider != "local" or not user.is_active:
            raise HTTPException(
                status_code=404,
                detail="가입된 계정을 찾을 수 없습니다."
            )

    auth_code = str(random.randint(100000, 999999))

    await redis_client.setex(
        get_email_auth_key(str(email)),
        settings.EMAIL_AUTH_TTL_SECONDS,
        auth_code,
    )

    if type == "reset-password":
        subject = "[Party-Up] 비밀번호 재설정 인증번호입니다."
        body = f"안녕하세요!\n\n비밀번호 재설정을 위한 인증번호는 [{auth_code}] 입니다."
    else:
        subject = "[Party-Up] 회원가입 인증번호입니다."
        body = f"안녕하세요!\n\n회원가입 인증번호는 [{auth_code}] 입니다."

    message = MessageSchema(
        subject=subject,
        recipients=[email],
        body=body,
        subtype="plain",
    )

    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, message)

    return {
        "message": "인증 메일이 발송되었습니다.",
        "expires_in": settings.EMAIL_AUTH_TTL_SECONDS,
    }

@router.post("/email-verify")
async def email_verify(email: str, code: str, type: str = "signup"):
    redis_key = get_email_auth_key(email)
    saved_code = await redis_client.get(redis_key)
    if not saved_code:
        raise HTTPException(status_code=400, detail="인증번호가 없거나 만료되었습니다.")
    if saved_code != code:
        raise HTTPException(status_code=400, detail="인증번호가 틀렸습니다.")
    await redis_client.delete(redis_key)

    if type == "reset-password":
        await redis_client.setex(
            f"password_reset_verified:{email}",
            600,
            "true",
        )

    return {"success": True, "message": "이메일 인증에 성공했습니다."}


@router.get("/users/check-nickname")
async def check_nickname(nickname: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.nickname == nickname))
    return {"exists": result.scalar_one_or_none() is not None}


@router.post("/users/find-id", response_model=FindIdResponse)
async def find_id(
    payload: FindIdRequest,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(User).where(
            User.nickname == payload.nickname,
            User.phone == payload.phone,
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        return FindIdResponse(message="일치하는 계정을 찾지 못했습니다.")

    return FindIdResponse(email=user.email)

@router.post("/users/find-password", response_model=FindPasswordResponse)
async def find_password(
    payload: FindPasswordRequest,
    db: AsyncSession = Depends(get_db),
):

    return FindPasswordResponse(
        message="입력하신 이메일로 비밀번호 재설정 안내를 진행할 수 있습니다."
    )


@router.post("/users/reset-password", response_model=ResetPasswordResponse)
async def reset_password(
    payload: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    verified_key = f"password_reset_verified:{payload.email}"
    verified = await redis_client.get(verified_key)

    if not verified:
        raise HTTPException(
            status_code=400,
            detail="비밀번호 재설정 인증이 완료되지 않았거나 만료되었습니다.",
        )

    result = await db.execute(
        select(User).where(User.email == payload.email)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=404,
            detail="가입된 계정을 찾을 수 없습니다.",
        )

    if user.provider != "local" or user.password_hash is None:
        raise HTTPException(
            status_code=400,
            detail="일반 로그인 계정만 비밀번호를 재설정할 수 있습니다.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="비활성화된 계정입니다.",
        )

    user.password_hash = get_password_hash(payload.new_password)
    await db.commit()

    await redis_client.delete(verified_key)

    return ResetPasswordResponse(
        message="비밀번호가 성공적으로 변경되었습니다."
    )


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
    client_ip = request.client.host if request.client else None
    if client_ip:
        is_ip_banned = await redis_client.get(f"ip:banned:{client_ip}")
        if is_ip_banned:
            raise HTTPException(status_code=403, detail="이용이 제한된 계정입니다.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="비활성화된 계정입니다.")

    await issue_tokens_and_save(
        response=response,
        db=db,
        user=user,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host if request.client else None,
    )

    await create_activity_log(
        db=db,
        user_id=user.id,
        action="LOGIN",
        description="로그인",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )

    return {
        "message": "로그인에 성공했습니다.",
        "user": {
            "email": user.email,
            "nickname": user.nickname,
            "role": user.role,
            "phone": user.phone,
        },
    }
