import requests
from fastapi import HTTPException, status

from core.config import settings


def _safe_json(response: requests.Response) -> dict:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}


def _raise_oauth_error(provider: str, message: str, response: requests.Response | None = None):
    detail = message
    if response is not None:
        detail = f"{message}: {_safe_json(response)}"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
    )


# ─── Google ──────────────────────────────────────────────────────
def get_google_access_token(code: str) -> str:
    if (
        not settings.GOOGLE_CLIENT_ID
        or not settings.GOOGLE_CLIENT_SECRET
        or not settings.GOOGLE_REDIRECT_URI
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="구글 OAuth 환경변수가 설정되지 않았습니다.",
        )

    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="구글 서버와 통신 중 오류가 발생했습니다.",
        )

    if response.status_code != 200:
        _raise_oauth_error("google", "구글 access token 발급 실패", response)

    data = _safe_json(response)
    access_token = data.get("access_token")
    if not access_token:
        _raise_oauth_error("google", "구글 access_token이 없습니다.", response)

    return access_token


async def get_google_user_info(access_token: str) -> dict:
    try:
        response = requests.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="구글 서버와 통신 중 오류가 발생했습니다.",
        )

    if response.status_code != 200:
        _raise_oauth_error("google", "유효하지 않은 구글 토큰입니다.", response)

    data = _safe_json(response)

    if not data.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="구글 사용자 식별값을 가져오지 못했습니다.",
        )

    return data


# ─── Kakao ───────────────────────────────────────────────────────

async def get_kakao_access_token(code: str) -> str:
    if not settings.KAKAO_REST_API_KEY or not settings.KAKAO_REDIRECT_URI:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="카카오 OAuth 환경변수가 설정되지 않았습니다.",
        )

    data = {
        "grant_type": "authorization_code",
        "client_id": settings.KAKAO_REST_API_KEY,
        "redirect_uri": settings.KAKAO_REDIRECT_URI,
        "code": code,
    }

    if getattr(settings, "KAKAO_CLIENT_SECRET", None):
        data["client_secret"] = settings.KAKAO_CLIENT_SECRET

    try:
        response = requests.post(
            "https://kauth.kakao.com/oauth/token",
            data=data,
            timeout=10,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="카카오 서버와 통신 중 오류가 발생했습니다.",
        )

    if response.status_code != 200:
        _raise_oauth_error("kakao", "카카오 access token 발급 실패", response)

    body = _safe_json(response)
    access_token = body.get("access_token")
    if not access_token:
        _raise_oauth_error("kakao", "카카오 access_token이 없습니다.", response)

    return access_token


async def get_kakao_user_info(access_token: str) -> dict:
    try:
        response = requests.get(
            "https://kapi.kakao.com/v2/user/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="카카오 서버와 통신 중 오류가 발생했습니다.",
        )

    if response.status_code != 200:
        _raise_oauth_error("kakao", "유효하지 않은 카카오 토큰입니다.", response)

    data = _safe_json(response)

    if not data.get("id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="카카오 사용자 식별값을 가져오지 못했습니다.",
        )

    return data


# ─── Naver ───────────────────────────────────────────────────────
def get_naver_access_token(code: str, state: str) -> str:
    if (
        not settings.NAVER_CLIENT_ID
        or not settings.NAVER_CLIENT_SECRET
        or not settings.NAVER_REDIRECT_URI
    ):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="네이버 OAuth 환경변수가 설정되지 않았습니다.",
        )

    try:
        response = requests.get(
            "https://nid.naver.com/oauth2.0/token",
            params={
                "grant_type": "authorization_code",
                "client_id": settings.NAVER_CLIENT_ID,
                "client_secret": settings.NAVER_CLIENT_SECRET,
                "code": code,
                "state": state,
            },
            timeout=10,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="네이버 서버와 통신 중 오류가 발생했습니다.",
        )

    data = _safe_json(response)

    if response.status_code != 200 or "access_token" not in data:
        _raise_oauth_error("naver", "네이버 access token 발급 실패", response)

    return data["access_token"]


async def get_naver_user_info(access_token: str) -> dict:
    try:
        response = requests.get(
            "https://openapi.naver.com/v1/nid/me",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="네이버 서버와 통신 중 오류가 발생했습니다.",
        )

    data = _safe_json(response)

    if response.status_code != 200 or "response" not in data:
        _raise_oauth_error("naver", "네이버 사용자 정보 조회 실패", response)

    user_info = data["response"]

    if not user_info.get("id"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="네이버 사용자 식별값을 가져오지 못했습니다.",
        )

    return user_info
