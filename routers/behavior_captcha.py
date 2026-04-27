from fastapi import APIRouter, Request
from fastapi.responses import Response

from schemas.captcha import (
    CaptchaChallengeResponse,
    CaptchaEnvInfo,
    CaptchaInitRequest,
    CaptchaInitResponse,
    CaptchaScreenInfo,
    CaptchaStatusResponse,
    CaptchaVerifyRequest,
    CaptchaVerifyResponse,
)
from services.captcha_service import (
    get_captcha_status,
    get_challenge,
    get_proxied_image,
    initiate_captcha,
    verify_challenge,
)

# 상원: 1차 행동 기반 캡챠 전용 라우터. handocr 라우터와 파일을 분리하여
#       서로의 경로/임포트가 간섭하지 않도록 격리합니다.
router = APIRouter(prefix="/captcha", tags=["BehaviorCaptcha"])


# ─────────────────────────────────────────────
# 1차 캡챠 API
# ─────────────────────────────────────────────


@router.post("/init", response_model=CaptchaInitResponse)  # 상원
async def captcha_init(payload: CaptchaInitRequest, request: Request):
    # 상원: 행동 데이터 기반 1차 판정을 시작하는 진입점입니다.
    # 상원: 실제 점수 계산과 세션 생성은 서비스 계층 함수 initiate_captcha에 위임합니다.
    return await initiate_captcha(payload, request)


@router.get("/challenge", response_model=CaptchaChallengeResponse)  # 상원
async def captcha_challenge(
    session_id: str,
    request: Request,
    force_refresh: bool = False,
):
    # 상원: challenge 상태 세션의 3x3 이미지 문제를 내려줍니다.
    # 상원: 세션 검증과 문제 생성은 서비스 계층 함수 get_challenge가 처리합니다.
    return await get_challenge(session_id, request, force_refresh=force_refresh)


@router.post("/verify", response_model=CaptchaVerifyResponse)  # 상원
async def captcha_verify(payload: CaptchaVerifyRequest, request: Request):
    # 상원: 사용자가 선택한 3칸이 이모지 순서와 맞는지 검증하고 통과 토큰을 발급합니다.
    # 상원: 실제 정답 판정과 토큰 발급은 서비스 계층 함수 verify_challenge가 처리합니다.
    return await verify_challenge(payload, request)


@router.get("/status", response_model=CaptchaStatusResponse)  # 상원
async def captcha_status(request: Request):
    # 상원: WAIT, LOCKED, BANNED와 active_session_id를 조회해 프론트 재진입 흐름을 맞춥니다.
    # 상원: 상태 계산은 서비스 계층 함수 get_captcha_status에 위임합니다.
    return await get_captcha_status(request)


@router.get("/image/{token}")
async def captcha_image_proxy(token: str):
    """이미지 프록시 — URL에서 동물 카테고리명 노출 방지"""
    image_bytes, content_type = await get_proxied_image(token)
    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={"Cache-Control": "no-store"},
    )


# ─────────────────────────────────────────────
# 발표 시연용 테스트 엔드포인트
# ─────────────────────────────────────────────


@router.post("/test/simulate-bot", response_model=CaptchaInitResponse)
async def simulate_bot(request: Request):
    """발표용: Selenium 봇 시뮬레이션 → block"""
    fake_payload = CaptchaInitRequest(
        mouse_moves=[],
        clicks=[],
        key_intervals=[],
        scrolled=False,
        env=CaptchaEnvInfo(
            webdriver=True,
            plugins_count=0,
            canvas_hash="",
            webgl_renderer="",
            screen=CaptchaScreenInfo(width=0, height=0),
            timezone="",
            languages=[],
        ),
        page_load_to_checkbox=50,
        trigger_type="new_ip_login",
    )
    return await initiate_captcha(fake_payload, request)
