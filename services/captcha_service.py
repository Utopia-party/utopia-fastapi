import hashlib
import json
import math
import random
import time
import uuid
from typing import Any
from urllib.parse import quote

from fastapi import HTTPException, Request, status
from jose import JWTError, jwt
from minio import Minio
from minio.error import S3Error

from core.config import settings
from core.redis_client import redis_client
from schemas.captcha import (
    CaptchaChallengeResponse,
    CaptchaEmojiItem,
    CaptchaInitRequest,
    CaptchaInitResponse,
    CaptchaPhotoItem,
    CaptchaStatusResponse,
    CaptchaVerifyRequest,
    CaptchaVerifyResponse,
)


# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────
# 1차 캡챠는 Redis 기반으로 동작시키고, 환경변수가 없더라도 로컬 개발이 가능하도록
# 대부분의 값을 안전한 기본값과 함께 둡니다.

CAPTCHA_PASS_THRESHOLD = getattr(settings, "CAPTCHA_PASS_THRESHOLD", 0.7)
CAPTCHA_CHALLENGE_THRESHOLD = getattr(settings, "CAPTCHA_CHALLENGE_THRESHOLD", 0.3)
CAPTCHA_SESSION_TTL_SECONDS = getattr(settings, "CAPTCHA_SESSION_TTL_SECONDS", 120)
CAPTCHA_TOKEN_TTL_SECONDS = getattr(settings, "CAPTCHA_TOKEN_TTL_SECONDS", 300)
CAPTCHA_TOKEN_MAX_USES = getattr(settings, "CAPTCHA_TOKEN_MAX_USES", 3)
CAPTCHA_MAX_ATTEMPTS = getattr(settings, "CAPTCHA_MAX_ATTEMPTS", 5)
CAPTCHA_LOCK_SECONDS = getattr(settings, "CAPTCHA_LOCK_SECONDS", 1800)
CAPTCHA_BAN_SECONDS = getattr(settings, "CAPTCHA_BAN_SECONDS", 86400)
CAPTCHA_WAIT_SECONDS = getattr(settings, "CAPTCHA_WAIT_SECONDS", 30)
CAPTCHA_RATE_LIMIT_WINDOW_SECONDS = getattr(
    settings,
    "CAPTCHA_RATE_LIMIT_WINDOW_SECONDS",
    60,
)
CAPTCHA_RATE_LIMIT_MAX_REQUESTS = getattr(
    settings,
    "CAPTCHA_RATE_LIMIT_MAX_REQUESTS",
    10,
)
CAPTCHA_MIN_SOLVE_SECONDS = getattr(settings, "CAPTCHA_MIN_SOLVE_SECONDS", 0.8)
CAPTCHA_JWT_SECRET = getattr(settings, "CAPTCHA_JWT_SECRET", "") or settings.SECRET_KEY
CAPTCHA_JWT_TYPE = "captcha"


# ─────────────────────────────────────────────
# 문제 세트 생성용 동물 이미지 메타데이터
# ─────────────────────────────────────────────

# 상원: FR-118 문제 세트 생성을 위해 captcha-emojis, captcha-photos 버킷에서 이미지를 읽어옵니다.
# 상원: 앱 시작 시 한 번만 로드해 challenge 생성 때 빠르게 재사용합니다.
minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
)

ANIMAL_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ANIMAL_LABELS: dict[str, str] = {
    "bear": "곰",
    "dog": "강아지",
    "fox": "여우",
    "penguin": "펭귄",
}
SUPPORTED_CHALLENGE_CATEGORIES = tuple(ANIMAL_LABELS.keys())


def _load_minio_asset_library(bucket: str) -> dict[str, list[str]]:
    """MinIO 버킷에서 카테고리별 이미지 목록 로드"""
    library: dict[str, list[str]] = {}
    try:
        objects = minio_client.list_objects(bucket, recursive=True)
        for obj in objects:
            name = obj.object_name  # e.g. "bear/bear_001.png" or "real_animal_photos/bear/photo.jpg"
            parts = name.split("/")
            # captcha-photos: real_animal_photos/bear/photo.jpg → category = bear
            # captcha-emojis: bear/bear_001.png → category = bear
            if len(parts) >= 2:
                if parts[0] == "real_animal_photos" and len(parts) >= 3:
                    category = parts[1].lower()
                else:
                    category = parts[0].lower()

                if category in SUPPORTED_CHALLENGE_CATEGORIES:
                    ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
                    if ext in ANIMAL_IMAGE_EXTENSIONS:
                        library.setdefault(category, []).append(name)
    except S3Error as e:
        print(f"[MinIO] 버킷 {bucket} 목록 조회 실패: {e}")
    return library


def _load_all_assets() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """이모지(GAN 생성)와 실사 사진 각각 로드"""
    emoji_lib = _load_minio_asset_library(settings.MINIO_EMOJI_BUCKET)
    photo_lib = _load_minio_asset_library(settings.MINIO_PHOTO_BUCKET)
    print(f"[MinIO] 이모지 로드: { {k: len(v) for k, v in emoji_lib.items()} }")
    print(f"[MinIO] 실사 로드: { {k: len(v) for k, v in photo_lib.items()} }")
    return emoji_lib, photo_lib


EMOJI_ASSET_LIBRARY, PHOTO_ASSET_LIBRARY = _load_all_assets()


# ─────────────────────────────────────────────
# Redis 키 생성 유틸
# ─────────────────────────────────────────────


def _session_key(session_id: str) -> str:
    return f"captcha:session:{session_id}"


def _token_key(jti: str) -> str:
    return f"captcha:token:{jti}"


def _rate_limit_key(client_ip: str) -> str:
    return f"rate:captcha:{client_ip}"


def _wait_key(client_ip: str) -> str:
    return f"captcha:wait:{client_ip}"


def _lock_key(client_ip: str) -> str:
    return f"captcha:lock:{client_ip}"


def _lock_count_key(client_ip: str) -> str:
    return f"captcha:lock-count:{client_ip}"


def _ban_key(client_ip: str) -> str:
    return f"captcha:ban:{client_ip}"


def _force_challenge_key(client_ip: str) -> str:
    return f"captcha:force-challenge:{client_ip}"


# ── 이번 수정: 진행 중 challenge 세션 추적용 Redis 키 ─────────
def _active_session_key(client_ip: str) -> str:
    return f"captcha:active-session:{client_ip}"


# ─────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────


def extract_client_ip(request: Request) -> str:
    # 프록시 뒤에 배치될 수 있으므로 X-Forwarded-For를 우선 확인합니다.
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def _now_ts() -> int:
    return int(time.time())


def _now_time() -> float:
    return time.time()


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def _serialize(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False)


async def _load_json(key: str) -> dict[str, Any] | None:
    raw_value = await redis_client.get(key)
    if not raw_value:
        return None
    return json.loads(raw_value)


async def _save_json(key: str, ttl_seconds: int, value: dict[str, Any]) -> None:
    await redis_client.setex(key, ttl_seconds, _serialize(value))


def _fingerprint_hash(payload: CaptchaInitRequest) -> str:
    # 해시에는 민감 정보 원문을 직접 남기지 않습니다.
    joined = "|".join(
        [
            payload.env.canvas_hash,
            payload.env.webgl_renderer,
            ",".join(payload.env.languages),
            payload.env.timezone,
            str(payload.env.screen.width),
            str(payload.env.screen.height),
        ]
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────
# 진행 중 challenge 세션 관리
# ─────────────────────────────────────────────
# 사용자가 모달을 닫거나 새로고침하더라도 "이미 시작한 challenge"는 같은 IP 기준으로
# 계속 이어서 풀게 만들기 위한 보조 키입니다.
# 이번 수정에서 추가한 핵심 구간입니다.


async def _bind_active_session(client_ip: str, session_id: str) -> None:
    await redis_client.setex(
        _active_session_key(client_ip),
        CAPTCHA_SESSION_TTL_SECONDS,
        session_id,
    )


async def _clear_active_session(client_ip: str) -> None:
    await redis_client.delete(_active_session_key(client_ip))


async def _get_active_session_id(client_ip: str) -> str | None:
    session_id = await redis_client.get(_active_session_key(client_ip))
    if not session_id:
        return None

    session = await _load_json(_session_key(session_id))
    if not session or session.get("status") != "challenge":
        await _clear_active_session(client_ip)
        return None

    return session_id


# ─────────────────────────────────────────────
# 상태 확인
# ─────────────────────────────────────────────


async def get_captcha_status(request: Request) -> CaptchaStatusResponse:
    client_ip = extract_client_ip(request)

    ban_ttl = await redis_client.ttl(_ban_key(client_ip))
    if ban_ttl > 0:
        return CaptchaStatusResponse(
            status="BANNED",
            message="반복 실패가 누적되어 접근이 차단되었습니다.",
            retry_after_seconds=ban_ttl,
        )

    lock_ttl = await redis_client.ttl(_lock_key(client_ip))
    if lock_ttl > 0:
        return CaptchaStatusResponse(
            status="LOCKED",
            message="실패 횟수 초과로 잠시 잠금 상태입니다.",
            retry_after_seconds=lock_ttl,
        )

    wait_ttl = await redis_client.ttl(_wait_key(client_ip))
    if wait_ttl > 0:
        return CaptchaStatusResponse(
            status="WAIT",
            message="요청이 많아 잠시 후 다시 시도해 주세요.",
            retry_after_seconds=wait_ttl,
        )

    active_session_id = await _get_active_session_id(client_ip)
    if active_session_id:
        return CaptchaStatusResponse(
            status="NORMAL",
            message="진행 중인 캡챠를 이어서 완료해 주세요.",
            active_session_id=active_session_id,
        )

    return CaptchaStatusResponse(
        status="NORMAL",
        message="캡챠 인증을 진행할 수 있습니다.",
    )


async def _ensure_security_state(request: Request) -> CaptchaStatusResponse | None:
    current_status = await get_captcha_status(request)
    if current_status.status == "NORMAL":
        return None
    return current_status


# ─────────────────────────────────────────────
# Rate Limit
# ─────────────────────────────────────────────


async def _check_rate_limit(client_ip: str) -> bool:
    # Redis Sorted Set으로 최근 1분 요청 횟수를 관리합니다.
    key = _rate_limit_key(client_ip)
    now_ms = int(time.time() * 1000)
    min_score = now_ms - (CAPTCHA_RATE_LIMIT_WINDOW_SECONDS * 1000)

    pipeline = redis_client.pipeline()
    pipeline.zremrangebyscore(key, 0, min_score)
    pipeline.zadd(key, {f"{now_ms}:{uuid.uuid4()}": now_ms})
    pipeline.zcard(key)
    pipeline.expire(key, CAPTCHA_RATE_LIMIT_WINDOW_SECONDS)
    _, _, request_count, _ = await pipeline.execute()

    return int(request_count) > CAPTCHA_RATE_LIMIT_MAX_REQUESTS


async def _mark_wait(client_ip: str) -> None:
    await redis_client.setex(_wait_key(client_ip), CAPTCHA_WAIT_SECONDS, "WAIT")


async def _mark_lock(client_ip: str) -> None:
    await redis_client.setex(_lock_key(client_ip), CAPTCHA_LOCK_SECONDS, "LOCKED")
    # 이번 수정: 잠금이 끝난 직후 첫 시도는 반드시 새 challenge로 돌려보내기 위해 플래그를 남깁니다.
    await redis_client.setex(
        _force_challenge_key(client_ip),
        max(CAPTCHA_LOCK_SECONDS + CAPTCHA_SESSION_TTL_SECONDS, 60),
        "FORCE_CHALLENGE",
    )

    lock_count_key = _lock_count_key(client_ip)
    current_count = await redis_client.incr(lock_count_key)
    await redis_client.expire(lock_count_key, CAPTCHA_BAN_SECONDS)

    # 같은 IP에서 잠금이 반복되면 하드 밴으로 올립니다.
    if int(current_count) >= 3:
        await redis_client.setex(_ban_key(client_ip), CAPTCHA_BAN_SECONDS, "BANNED")


# ─────────────────────────────────────────────
# 점수 계산
# ─────────────────────────────────────────────


def _calculate_mouse_score(payload: CaptchaInitRequest) -> float:
    moves = payload.mouse_moves
    if len(moves) < 4:
        return 0.08

    distances: list[float] = []
    speeds: list[float] = []
    direction_changes = 0
    previous_angle: float | None = None

    for previous, current in zip(moves, moves[1:]):
        delta_x = current.x - previous.x
        delta_y = current.y - previous.y
        delta_t = max(current.t - previous.t, 1)
        distance = math.hypot(delta_x, delta_y)
        speed = distance / delta_t

        distances.append(distance)
        speeds.append(speed)

        current_angle = math.atan2(delta_y, delta_x)
        if previous_angle is not None and abs(current_angle - previous_angle) > 0.55:
            direction_changes += 1
        previous_angle = current_angle

    total_distance = sum(distances)
    straight_distance = math.hypot(
        moves[-1].x - moves[0].x,
        moves[-1].y - moves[0].y,
    )
    directness = straight_distance / max(total_distance, 1.0)

    speed_mean = sum(speeds) / max(len(speeds), 1)
    speed_variance = math.sqrt(_variance(speeds))

    point_density_score = _clamp(len(moves) / 60)
    coverage_score = _clamp(total_distance / 900)
    direction_score = _clamp(direction_changes / 12)
    variance_score = _clamp(speed_variance / max(speed_mean * 3, 0.05))
    directness_score = 1.0 - _clamp((directness - 0.82) / 0.18)

    return _clamp(
        (point_density_score * 0.2)
        + (coverage_score * 0.25)
        + (direction_score * 0.2)
        + (variance_score * 0.2)
        + (directness_score * 0.15)
    )


def _calculate_click_score(payload: CaptchaInitRequest) -> float:
    clicks = payload.clicks
    if not clicks:
        return 0.45

    x_values = [click.x for click in clicks]
    y_values = [click.y for click in clicks]
    click_intervals = [
        max(current.t - previous.t, 1)
        for previous, current in zip(clicks, clicks[1:])
    ]

    spread_x = math.sqrt(_variance(x_values))
    spread_y = math.sqrt(_variance(y_values))
    interval_variance = math.sqrt(_variance([float(value) for value in click_intervals]))
    move_to_click_ratio = len(payload.mouse_moves) / max(len(clicks), 1)

    spread_score = _clamp((spread_x + spread_y) / 180)
    interval_score = _clamp(interval_variance / 250) if click_intervals else 0.45

    if move_to_click_ratio < 2:
        ratio_score = 0.12
    elif move_to_click_ratio < 5:
        ratio_score = 0.4
    else:
        ratio_score = _clamp(move_to_click_ratio / 18)

    click_count_score = _clamp(len(clicks) / 3)

    return _clamp(
        (spread_score * 0.35)
        + (interval_score * 0.25)
        + (ratio_score * 0.25)
        + (click_count_score * 0.15)
    )


def _calculate_timing_score(payload: CaptchaInitRequest) -> float:
    # key_intervals는 "이전 키 입력과의 간격"이므로 첫 액션 시각 계산에는 직접 쓰지 않습니다.
    event_times = [move.t for move in payload.mouse_moves]
    event_times.extend(click.t for click in payload.clicks)

    first_action_delay = min(event_times) if event_times else payload.page_load_to_checkbox
    checkbox_delay = payload.page_load_to_checkbox

    if first_action_delay < 100:
        first_action_score = 0.0
    elif first_action_delay < 300:
        first_action_score = 0.2
    elif first_action_delay < 700:
        first_action_score = 0.55
    elif first_action_delay < 1200:
        first_action_score = 0.8
    else:
        first_action_score = 1.0

    checkbox_delay_score = _clamp(checkbox_delay / 2500)

    key_variance_score = (
        _clamp(math.sqrt(_variance([float(value) for value in payload.key_intervals])) / 180)
        if payload.key_intervals
        else 0.45
    )

    scroll_score = 0.65 if payload.scrolled else 0.35

    return _clamp(
        (first_action_score * 0.4)
        + (checkbox_delay_score * 0.3)
        + (key_variance_score * 0.2)
        + (scroll_score * 0.1)
    )


def _evaluate_environment(payload: CaptchaInitRequest) -> tuple[float, bool]:
    env = payload.env

    # webdriver=true는 Selenium/Puppeteer 자동화를 강하게 의심할 수 있는 신호입니다.
    if env.webdriver:
        return 0.0, True

    score = 0.0
    score += 0.2 if env.plugins_count > 0 else 0.04
    score += 0.2 if env.canvas_hash not in {"", "no-canvas", "canvas-error"} else 0.04
    score += 0.2 if env.webgl_renderer not in {"", "no-webgl", "webgl-error"} else 0.04
    score += 0.15 if env.languages else 0.05
    score += 0.1 if env.timezone else 0.0
    score += 0.15 if env.screen.width > 0 and env.screen.height > 0 else 0.0

    return _clamp(score), False


def _evaluate_headers(request: Request) -> tuple[float, bool]:
    headers = request.headers
    user_agent = headers.get("user-agent", "")
    accept = headers.get("accept", "")
    accept_language = headers.get("accept-language", "")
    sec_fetch_site = headers.get("sec-fetch-site", "")
    sec_fetch_mode = headers.get("sec-fetch-mode", "")
    sec_fetch_dest = headers.get("sec-fetch-dest", "")

    lowered_ua = user_agent.lower()

    # requests, curl, Go http-client 등 브라우저가 아닌 클라이언트는 즉시 차단합니다.
    if any(keyword in lowered_ua for keyword in ("python-requests", "curl/", "go-http-client")):
        return 0.0, True

    score = 0.0
    score += 0.25 if user_agent else 0.0
    score += 0.2 if accept else 0.0
    score += 0.2 if accept_language else 0.0
    score += 0.12 if sec_fetch_site else 0.0
    score += 0.12 if sec_fetch_mode else 0.0
    score += 0.11 if sec_fetch_dest else 0.0

    # 브라우저를 주장하는데 fetch 관련 헤더가 전혀 없다면 점수를 깎습니다.
    if "mozilla" in lowered_ua and not all((sec_fetch_site, sec_fetch_mode, sec_fetch_dest)):
        score -= 0.15

    return _clamp(score), False


def _calculate_scores(payload: CaptchaInitRequest, request: Request) -> tuple[float, float, float, bool]:
    mouse_score = _calculate_mouse_score(payload)
    click_score = _calculate_click_score(payload)
    timing_score = _calculate_timing_score(payload)
    behavior_score = _clamp((mouse_score * 0.35) + (click_score * 0.25) + (timing_score * 0.4))

    environment_score, env_block = _evaluate_environment(payload)
    header_score, header_block = _evaluate_headers(request)

    final_score = _clamp(
        (behavior_score * 0.6) + (environment_score * 0.2) + (header_score * 0.2)
    )

    return behavior_score, environment_score, final_score, env_block or header_block


# ─────────────────────────────────────────────
# 문제 생성
# ─────────────────────────────────────────────


# 상원: 브라우저가 captcha-emojis, captcha-photos 버킷 이미지를 바로 읽을 수 있도록 공개 URL을 만듭니다.
def _build_minio_url(bucket: str, object_name: str) -> str:
    """MinIO 오브젝트의 직접 접근 URL 생성 (브라우저에서 접근 가능한 공개 URL)"""
    public_endpoint = getattr(settings, "MINIO_PUBLIC_ENDPOINT", None) or settings.MINIO_ENDPOINT
    protocol = "https" if settings.MINIO_SECURE else "http"
    return f"{protocol}://{public_endpoint}/{bucket}/{quote(object_name, safe='/')}"


def _pick_from_library(library: dict[str, list[str]], category: str, used_paths: set[str]) -> str:
    candidates = [path for path in library[category] if path not in used_paths]
    if not candidates:
        candidates = library[category]
    selected = random.choice(candidates)
    used_paths.add(selected)
    return selected


def _build_new_challenge_payload(
    request: Request,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    # 이모지가 없는 카테고리는 실사로 대체 (GAN 생성 전 임시)
    emoji_lib = EMOJI_ASSET_LIBRARY if EMOJI_ASSET_LIBRARY else PHOTO_ASSET_LIBRARY
    photo_lib = PHOTO_ASSET_LIBRARY

    # 이모지가 없는 카테고리는 실사 이미지로 대체 (임시)
    available_categories = [
        category
        for category in SUPPORTED_CHALLENGE_CATEGORIES
        if photo_lib.get(category)
    ]

    if len(available_categories) < 3:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"MinIO에 캡챠용 동물 이미지가 부족합니다. "
                f"사용 가능: {available_categories} / 필요: 최소 3개 카테고리"
            ),
        )

    categories = random.sample(available_categories, 3)
    wrong_categories = [category for category in available_categories if category not in categories]
    if not wrong_categories:
        wrong_categories = [c for c in available_categories if c != categories[0]]
    answer_positions = sorted(random.sample(list(range(9)), 3))
    used_emoji_paths: set[str] = set()
    used_photo_paths: set[str] = set()

    emoji_bucket = settings.MINIO_EMOJI_BUCKET if EMOJI_ASSET_LIBRARY else settings.MINIO_PHOTO_BUCKET

    emojis: list[dict[str, Any]] = []
    for index, category in enumerate(categories):
        # 이모지가 없는 카테고리는 실사 이미지로 대체 (임시)
        if emoji_lib.get(category):
            asset_path = _pick_from_library(emoji_lib, category, used_emoji_paths)
            bucket = emoji_bucket
        else:
            asset_path = _pick_from_library(photo_lib, category, used_photo_paths)
            bucket = settings.MINIO_PHOTO_BUCKET
        emojis.append(
            {
                "id": f"emoji-{category}-{index}",
                "url": _build_minio_url(bucket, asset_path),
                "category": category,
            }
        )

    photos: list[dict[str, Any]] = []
    correct_index = 0
    for position in range(9):
        if position in answer_positions:
            category = categories[correct_index]
            correct_index += 1
        else:
            category = random.choice(wrong_categories)

        asset_path = _pick_from_library(photo_lib, category, used_photo_paths)
        photos.append(
            {
                "id": f"photo-{category}-{position}",
                "url": _build_minio_url(settings.MINIO_PHOTO_BUCKET, asset_path),
                "index": position,
                # 상원: verify 단계에서 칸 번호를 다시 동물 카테고리로 복원하려고 사진에도 category를 넣어둡니다.
                "category": category,  # 상원
            }
        )

    return emojis, photos, answer_positions


# ─────────────────────────────────────────────
# 토큰 발급 / 검증
# ─────────────────────────────────────────────


async def _issue_captcha_token(client_ip: str, score: float, session_id: str) -> str:
    now_ts = _now_ts()
    expires_at = now_ts + CAPTCHA_TOKEN_TTL_SECONDS
    jti = str(uuid.uuid4())

    payload = {
        "sub": session_id,
        "jti": jti,
        "ip": client_ip,
        "score": round(score, 4),
        "type": CAPTCHA_JWT_TYPE,
        "iat": now_ts,
        "exp": expires_at,
    }

    await redis_client.setex(
        _token_key(jti),
        CAPTCHA_TOKEN_TTL_SECONDS,
        _serialize({"uses_left": CAPTCHA_TOKEN_MAX_USES, "score": score, "ip": client_ip}),
    )

    return jwt.encode(payload, CAPTCHA_JWT_SECRET, algorithm=settings.ALGORITHM)


async def validate_captcha_token(request: Request) -> dict[str, Any]:
    raw_token = request.headers.get("X-Captcha-Token")
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="캡챠 인증이 필요합니다.",
        )

    try:
        payload = jwt.decode(
            raw_token,
            CAPTCHA_JWT_SECRET,
            algorithms=[settings.ALGORITHM],
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="유효하지 않은 캡챠 토큰입니다.",
        ) from exc

    if payload.get("type") != CAPTCHA_JWT_TYPE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="캡챠 토큰 형식이 올바르지 않습니다.",
        )

    client_ip = extract_client_ip(request)
    if payload.get("ip") != client_ip:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="캡챠 토큰의 클라이언트 정보가 일치하지 않습니다.",
        )

    jti = payload.get("jti")
    if not jti:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="캡챠 토큰 식별자가 없습니다.",
        )

    token_state = await _load_json(_token_key(jti))
    if not token_state:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="캡챠 토큰이 만료되었거나 이미 사용할 수 없습니다.",
        )

    uses_left = int(token_state.get("uses_left", 0))
    if uses_left <= 0:
        await redis_client.delete(_token_key(jti))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="캡챠 토큰의 사용 가능 횟수를 초과했습니다.",
        )

    # 로그인/회원가입을 몇 번 다시 시도할 수 있도록 사용 횟수를 줄여가며 검증합니다.
    token_state["uses_left"] = uses_left - 1
    await _save_json(_token_key(jti), CAPTCHA_TOKEN_TTL_SECONDS, token_state)

    return payload


# ─────────────────────────────────────────────
# 공개 서비스 함수
# ─────────────────────────────────────────────


async def initiate_captcha(payload: CaptchaInitRequest, request: Request) -> CaptchaInitResponse:
    current_status = await _ensure_security_state(request)
    if current_status:
        return CaptchaInitResponse(status="block", message=current_status.message)

    client_ip = extract_client_ip(request)

    # 이번 수정: 이미 시작한 challenge가 남아 있으면 새 캡챠를 만들지 않고 같은 세션으로 복귀시킵니다.
    # 이미 시작한 challenge가 있으면 새 판정을 하지 않고 같은 세션으로 되돌립니다.
    active_session_id = await _get_active_session_id(client_ip)
    if active_session_id:
        return CaptchaInitResponse(status="challenge", session_id=active_session_id)

    if await _check_rate_limit(client_ip):
        await _mark_wait(client_ip)
        return CaptchaInitResponse(
            status="block",
            message="요청이 많아 잠시 후 다시 시도해 주세요.",
        )

    # 이번 수정: 잠금 해제 후 첫 재시도는 pass로 빠지지 않고 반드시 challenge를 다시 시작합니다.
    # 잠금이 해제된 직후 첫 시도는 행동 점수와 무관하게 반드시 이미지 캡챠로 다시 보냅니다.
    if await redis_client.get(_force_challenge_key(client_ip)):
        await redis_client.delete(_force_challenge_key(client_ip))

        session_id = str(uuid.uuid4())
        session_payload = {
            "session_id": session_id,
            "client_ip": client_ip,
            "trigger_type": payload.trigger_type,
            "client_session_id": payload.session_id,
            "fingerprint_hash": _fingerprint_hash(payload),
            "behavior_score": 0.0,
            "environment_score": 0.0,
            "final_score": 0.0,
            "attempts": 0,
            "status": "challenge",
            "created_at": _now_time(),
            "forced_after_lock": True,
        }
        await _save_json(_session_key(session_id), CAPTCHA_SESSION_TTL_SECONDS, session_payload)
        await _bind_active_session(client_ip, session_id)

        return CaptchaInitResponse(status="challenge", session_id=session_id)

    behavior_score, environment_score, final_score, immediate_block = _calculate_scores(
        payload,
        request,
    )
    if immediate_block:
        return CaptchaInitResponse(
            status="block",
            message="비정상적인 브라우저 환경이 감지되었습니다.",
        )

    if final_score >= CAPTCHA_PASS_THRESHOLD:
        token = await _issue_captcha_token(
            client_ip,
            final_score,
            payload.session_id or str(uuid.uuid4()),
        )
        return CaptchaInitResponse(status="pass", token=token)

    if final_score <= CAPTCHA_CHALLENGE_THRESHOLD:
        return CaptchaInitResponse(
            status="block",
            message="보안 정책에 따라 접근이 제한되었습니다.",
        )

    session_id = str(uuid.uuid4())
    session_payload = {
        "session_id": session_id,
        "client_ip": client_ip,
        "trigger_type": payload.trigger_type,
        "client_session_id": payload.session_id,
        "fingerprint_hash": _fingerprint_hash(payload),
        "behavior_score": round(behavior_score, 4),
        "environment_score": round(environment_score, 4),
        "final_score": round(final_score, 4),
        "attempts": 0,
        "status": "challenge",
        "created_at": _now_time(),
    }
    await _save_json(_session_key(session_id), CAPTCHA_SESSION_TTL_SECONDS, session_payload)
    await _bind_active_session(client_ip, session_id)

    return CaptchaInitResponse(status="challenge", session_id=session_id)


# ── 수정됨: get_challenge에 디버그 print 추가 ─────────────────
async def get_challenge(session_id: str, request: Request) -> CaptchaChallengeResponse:
    session = await _load_json(_session_key(session_id))
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="캡챠 세션이 만료되었습니다.",
        )

    client_ip = extract_client_ip(request)
    session_ip = session.get("client_ip")
    print(f"[DEBUG] get_challenge: session_ip={session_ip}, request_ip={client_ip}")
    print(f"[DEBUG] Headers: {dict(request.headers)}")

    if session_ip != client_ip:
        print(f"[DEBUG] IP MISMATCH!")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"IP 불일치: session={session_ip}, request={client_ip}",
        )

    if session.get("status") != "challenge":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미지 캡챠를 진행할 수 없는 세션 상태입니다.",
        )

    await _bind_active_session(client_ip, session_id)

    if not session.get("emojis") or not session.get("photos"):
        emojis, photos, answer_indices = _build_new_challenge_payload(request)
        session["emojis"] = emojis
        session["photos"] = photos
        session["answer_indices"] = answer_indices
        session["challenge_issued_at"] = _now_time()
        await _save_json(_session_key(session_id), CAPTCHA_SESSION_TTL_SECONDS, session)

    return CaptchaChallengeResponse(
        session_id=session_id,
        emojis=[CaptchaEmojiItem(**item) for item in session["emojis"]],
        photos=[CaptchaPhotoItem(**item) for item in session["photos"]],
    )


async def verify_challenge(payload: CaptchaVerifyRequest, request: Request) -> CaptchaVerifyResponse:
    session = await _load_json(_session_key(payload.session_id))
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="캡챠 세션이 만료되었습니다.",
        )

    client_ip = extract_client_ip(request)
    if session.get("client_ip") != client_ip:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="세션을 발급받은 클라이언트와 현재 요청이 일치하지 않습니다.",
        )

    issued_at = float(session.get("challenge_issued_at", session.get("created_at", _now_time())))
    solve_seconds = max(_now_time() - issued_at, 0.0)

    # 상원: 현재 세션이 몇 번째 시도인지 계산해 남은 횟수 응답에 사용합니다.
    attempts = int(session.get("attempts", 0))
    # 상원: 기존 answer_indices는 디버그용으로 남기고, 실제 정답 판정은 이모지의 동물 순서 기준으로 다시 계산합니다.
    answer_indices = session.get("answer_indices", [])  # 상원
    # 상원: 위쪽 이모지 3개의 동물 순서를 추출해 사용자가 맞춰야 할 정답 순서를 만듭니다.
    expected_categories = [emoji.get("category") for emoji in session.get("emojis", [])]  # 상원
    # 상원: 아래 사진 9칸의 index를 실제 동물 category와 연결하는 사전을 만듭니다.
    photo_category_by_index = {  # 상원
        int(photo.get("index")): photo.get("category")  # 상원
        for photo in session.get("photos", [])  # 상원
    }  # 상원
    # 상원: 사용자가 누른 칸 번호를 실제 동물 카테고리 순서로 바꿔서 화면 문구와 같은 기준으로 검증합니다.
    selected_categories = [  # 상원
        photo_category_by_index.get(selected_index)  # 상원
        for selected_index in payload.selected_indices  # 상원
    ]  # 상원

    # 상원: 중복 클릭을 막고, 선택한 동물 순서가 이모지 순서와 완전히 같을 때만 정답 처리합니다.
    is_correct = (  # 상원
        # 상원: 먼저 사용자가 정확히 3칸을 골랐는지 길이를 검사합니다.
        len(payload.selected_indices) == len(expected_categories)  # 상원
        # 상원: 같은 칸을 두 번 누른 경우는 정답으로 인정하지 않으려고 중복을 막습니다.
        and len(payload.selected_indices) == len(set(payload.selected_indices))  # 상원
        # 상원: 최종적으로 사용자가 고른 동물 순서가 이모지 정답 순서와 완전히 같은지 비교합니다.
        and selected_categories == expected_categories  # 상원
    )  # 상원

    # 상원: 실제 서버가 무엇을 정답으로 봤는지 콘솔에서 바로 비교할 수 있도록 디버그 로그를 남깁니다.
    print(  # 상원
        "[DEBUG] verify_challenge:",  # 상원
        {  # 상원
            "selected_indices": payload.selected_indices,  # 상원
            "answer_indices": answer_indices,  # 상원
            "selected_categories": selected_categories,  # 상원
            "expected_categories": expected_categories,  # 상원
        },  # 상원
    )  # 상원

    # 상원: 사람이 아니라 스크립트로 너무 빠르게 푸는 경우를 막으려고 최소 풀이 시간도 함께 검사합니다.
    if solve_seconds < CAPTCHA_MIN_SOLVE_SECONDS:
        is_correct = False

    # 상원: 정답이면 캡챠 통과 토큰을 발급하고 세션을 삭제한 뒤 성공 응답을 반환합니다.
    if is_correct:
        token = await _issue_captcha_token(
            client_ip,
            float(session.get("final_score", 0.5)),
            payload.session_id,
        )
        await redis_client.delete(_session_key(payload.session_id))
        await _clear_active_session(client_ip)
        return CaptchaVerifyResponse(success=True, token=token)

    attempts += 1
    remaining_attempts = max(CAPTCHA_MAX_ATTEMPTS - attempts, 0)

    if remaining_attempts == 0:
        session["attempts"] = attempts
        session["status"] = "blocked"
        await _save_json(_session_key(payload.session_id), CAPTCHA_SESSION_TTL_SECONDS, session)
        await _clear_active_session(client_ip)
        await _mark_lock(client_ip)
        return CaptchaVerifyResponse(
            success=False,
            remaining_attempts=0,
            message="실패 횟수를 초과해 잠금 상태로 전환되었습니다.",
        )

    # 오답일 때는 다음 challenge 요청에서 새 문제를 만들 수 있도록 기존 문제를 비웁니다.
    session["attempts"] = attempts
    session["emojis"] = []
    session["photos"] = []
    session["answer_indices"] = []
    session["challenge_issued_at"] = _now_time()
    await _save_json(_session_key(payload.session_id), CAPTCHA_SESSION_TTL_SECONDS, session)
    await _bind_active_session(client_ip, payload.session_id)

    if solve_seconds < CAPTCHA_MIN_SOLVE_SECONDS:
        message = "풀이 시간이 너무 짧아 자동 실패 처리되었습니다."
    else:
        message = "정답이 아닙니다. 새로운 문제로 다시 시도해 주세요."

    return CaptchaVerifyResponse(
        success=False,
        remaining_attempts=remaining_attempts,
        message=message,
    )
