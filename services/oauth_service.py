import httpx
from fastapi import HTTPException, status
from core.config import settings


# ─── Google ──────────────────────────────────────────────────────

async def get_google_access_token(code: str) -> str:
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET or not settings.GOOGLE_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="구글 OAuth 환경변수가 설정되지 않았습니다.")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="구글 서버와 통신 중 오류가 발생했습니다.")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail=f"구글 access token 발급 실패: {resp.json()}")
    access_token = resp.json().get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="구글 access_token이 없습니다.")
    return access_token


async def get_google_user_info(access_token: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://openidconnect.googleapis.com/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="구글 서버와 통신 중 오류가 발생했습니다.")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="유효하지 않은 구글 토큰입니다.")
    return resp.json()


# ─── Kakao ───────────────────────────────────────────────────────

async def get_kakao_access_token(code: str) -> str:
    if not settings.KAKAO_REST_API_KEY or not settings.KAKAO_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="카카오 OAuth 환경변수가 설정되지 않았습니다.")

    data = {
        "grant_type": "authorization_code",
        "client_id": settings.KAKAO_REST_API_KEY,
        "redirect_uri": settings.KAKAO_REDIRECT_URI,
        "code": code,
    }
    if settings.KAKAO_CLIENT_SECRET:
        data["client_secret"] = settings.KAKAO_CLIENT_SECRET

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("https://kauth.kakao.com/oauth/token", data=data)
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="카카오 서버와 통신 중 오류가 발생했습니다.")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail=f"카카오 access token 발급 실패: {resp.json()}")
    access_token = resp.json().get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="카카오 access_token이 없습니다.")
    return access_token


async def get_kakao_user_info(access_token: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://kapi.kakao.com/v2/user/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="카카오 서버와 통신 중 오류가 발생했습니다.")

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="유효하지 않은 카카오 토큰입니다.")
    return resp.json()


# ─── Naver ───────────────────────────────────────────────────────

async def get_naver_access_token(code: str, state: str) -> str:
    if not settings.NAVER_CLIENT_ID or not settings.NAVER_CLIENT_SECRET or not settings.NAVER_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="네이버 OAuth 환경변수가 설정되지 않았습니다.")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://nid.naver.com/oauth2.0/token",
                params={
                    "grant_type": "authorization_code",
                    "client_id": settings.NAVER_CLIENT_ID,
                    "client_secret": settings.NAVER_CLIENT_SECRET,
                    "code": code,
                    "state": state,
                    "redirect_uri": settings.NAVER_REDIRECT_URI,
                },
            )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="네이버 서버와 통신 중 오류가 발생했습니다.")

    data = resp.json()
    if resp.status_code != 200 or "access_token" not in data:
        raise HTTPException(status_code=401, detail=f"네이버 access token 발급 실패: {data}")
    return data["access_token"]


async def get_naver_user_info(access_token: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://openapi.naver.com/v1/nid/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="네이버 서버와 통신 중 오류가 발생했습니다.")

    data = resp.json()
    if resp.status_code != 200 or "response" not in data:
        raise HTTPException(status_code=401, detail=f"네이버 사용자 정보 조회 실패: {data}")
    return data["response"]
