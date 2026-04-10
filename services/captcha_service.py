import asyncio
import hashlib
import json
import logging
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
from sqlalchemy import text

from core.config import settings
from core.database import AsyncSessionLocal
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

logger = logging.getLogger("captcha_service")


_OPTIONAL_CAPTCHA_TABLES: dict[str, bool] = {
    "captcha_sets": True,
    "captcha_sessions": True,
    "behavior_embeddings": True,
}


def _is_missing_relation_error(exc: Exception, table_name: str) -> bool:
    message = str(exc).lower()
    return (
        "undefinedtableerror" in message
        or (f'relation "{table_name.lower()}" does not exist' in message)
        or (f"relation '{table_name.lower()}' does not exist" in message)
    )


def _disable_optional_table(table_name: str, exc: Exception, context: str) -> None:
    if not _OPTIONAL_CAPTCHA_TABLES.get(table_name, True):
        return
    _OPTIONAL_CAPTCHA_TABLES[table_name] = False
    logger.warning(
        "[DB] %s unavailable during %s; disabling optional DB feature: %s",
        table_name,
        context,
        exc,
    )


def _optional_table_enabled(table_name: str) -> bool:
    return _OPTIONAL_CAPTCHA_TABLES.get(table_name, True)


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

# 이미지 바이트 인메모리 캐시 (LRU)
# 캡챠 이미지는 고정 풀에서 반복 사용되므로 한 번 받아오면 영구 캐시.
# 키: (bucket, key) → value: (bytes, content_type)
# 약 2000장 × 평균 100KB = 약 200MB 상한 가정.
_IMAGE_CACHE_MAX = 2000
_image_cache: dict[tuple[str, str], tuple[bytes, str]] = {}


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
    "cat": "고양이",
    "dog": "강아지",
    "elephant": "코끼리",
    "fox": "여우",
    "horse": "말",
    "lion": "사자",
    "penguin": "펭귄",
    "tiger": "호랑이",
    "wolf": "늑대",
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


def _calculate_fingerprint_score(payload: CaptchaInitRequest) -> float:
    """md 스펙 레이어3의 핑거프린트 20% 가중치용 점수.
    브라우저 핑거프린트(canvas/webgl/플러그인/언어/시간대/화면) 안정성을 0~1로 평가.
    """
    env = payload.env
    score = 0.0
    if env.canvas_hash and env.canvas_hash not in {"", "no-canvas", "canvas-error"}:
        score += 0.25
    if env.webgl_renderer and env.webgl_renderer not in {"", "no-webgl", "webgl-error"}:
        score += 0.25
    if env.plugins_count > 0:
        score += 0.15
    if env.languages:
        score += 0.10
    if env.timezone:
        score += 0.10
    if env.screen.width > 0 and env.screen.height > 0:
        score += 0.15
    return _clamp(score)


async def _calculate_scores(
    payload: CaptchaInitRequest,
    request: Request,
    behavior_vector: list[float],
) -> tuple[float, float, float, float, float, bool]:
    """md 스펙 비간섭 검증 5겹 점수 계산.

    Returns:
        (rule_score, vector_score, environment_score, header_score, fingerprint_score, final_score, immediate_block)

    - 레이어3(룰): 마우스35 + 클릭25 + 타이밍20 + 핑거프린트20
    - 레이어4(벡터 KNN): pgvector 코사인 Top-5 → human_ratio 기반 점수
    - 최종: rule×0.5 + vector×0.5  (md 스펙)
    - 레이어1·2는 즉시 차단 게이트 역할 (점수에 직접 합산하지 않음)
    """
    mouse_score = _calculate_mouse_score(payload)
    click_score = _calculate_click_score(payload)
    timing_score = _calculate_timing_score(payload)
    fingerprint_score = _calculate_fingerprint_score(payload)

    # 레이어3: 룰 기반 점수 (md 스펙 가중치 복구)
    rule_score = _clamp(
        (mouse_score * 0.35)
        + (click_score * 0.25)
        + (timing_score * 0.20)
        + (fingerprint_score * 0.20)
    )

    # 레이어4: pgvector KNN (15차원 → K=5)
    similar = await _search_similar_behaviors(behavior_vector, top_k=5)
    vector_score = await _calculate_vector_score(similar)

    # 레이어1·2: 환경/헤더 게이트 (즉시 차단만, 최종 점수엔 직접 반영 X)
    environment_score, env_block = _evaluate_environment(payload)
    header_score, header_block = _evaluate_headers(request)

    # md 스펙: 룰×0.5 + 벡터×0.5
    # Cold Start 보호: 벡터 데이터가 부족하면 룰 가중치를 높여 중립 값에 휩쓸리지 않게.
    sample_size = len(similar)
    if sample_size >= 5:
        vector_weight = 0.5
    elif sample_size >= 3:
        vector_weight = 0.3
    else:
        vector_weight = 0.1  # 초기: 룰 기반에 거의 의존
    final_score = _clamp(
        (rule_score * (1.0 - vector_weight)) + (vector_score * vector_weight)
    )

    logger.info(
        f"[captcha.score] mouse={mouse_score:.2f} click={click_score:.2f} "
        f"timing={timing_score:.2f} fp={fingerprint_score:.2f} "
        f"rule={rule_score:.2f} vector={vector_score:.2f}(n={sample_size}) "
        f"env={environment_score:.2f} hdr={header_score:.2f} "
        f"final={final_score:.2f}"
    )

    return (
        rule_score,
        vector_score,
        environment_score,
        header_score,
        fingerprint_score,
        final_score,
        env_block or header_block,
    )


# ─────────────────────────────────────────────
# 문제 생성
# ─────────────────────────────────────────────


def _build_minio_url(bucket: str, object_name: str) -> str:
    """MinIO 오브젝트의 직접 접근 URL 생성 (서버 내부용)"""
    endpoint = settings.MINIO_ENDPOINT
    protocol = "https" if settings.MINIO_SECURE else "http"
    return f"{protocol}://{public_endpoint}/{bucket}/{quote(object_name, safe='/')}"


# ─────────────────────────────────────────────
# 이미지 프록시 (URL에서 동물명 노출 방지)
# ─────────────────────────────────────────────

IMAGE_TOKEN_TTL = 300  # 5분


async def _create_image_token(bucket: str, object_name: str) -> str:
    """이미지에 랜덤 토큰 부여, Redis에 매핑 저장"""
    token = uuid.uuid4().hex[:16]
    await redis_client.setex(
        f"captcha:img:{token}",
        IMAGE_TOKEN_TTL,
        json.dumps({"bucket": bucket, "key": object_name}),
    )
    return token


def _build_proxy_url(token: str) -> str:
    """프록시 URL 생성 — 브라우저에 노출되는 URL"""
    return f"/api/captcha/image/{token}"


def _minio_fetch_sync(bucket: str, key: str) -> bytes:
    """동기 MinIO 호출 (asyncio.to_thread에서 실행)"""
    response = minio_client.get_object(bucket, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _content_type_for(key: str) -> str:
    ext = key.rsplit(".", 1)[-1].lower() if "." in key else "png"
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "webp": "image/webp",
    }.get(ext, "image/png")


async def get_proxied_image(token: str) -> tuple[bytes, str]:
    """토큰으로 Redis에서 실제 경로 조회 → 캐시 또는 MinIO에서 이미지 반환.
    캡챠 이미지는 고정 풀에서 반복 사용되므로 인메모리 캐시 적중률이 매우 높음.
    """
    raw = await redis_client.get(f"captcha:img:{token}")
    if not raw:
        raise HTTPException(status_code=404, detail="이미지 토큰이 만료되었습니다.")

    info = json.loads(raw)
    bucket = info["bucket"]
    key = info["key"]

    # 1차: 인메모리 캐시 hit → MinIO 호출 0
    cached = _image_cache.get((bucket, key))
    if cached is not None:
        return cached

    # 2차: MinIO 호출 (스레드풀로 분리해 이벤트 루프 블록 방지)
    try:
        image_bytes = await asyncio.to_thread(_minio_fetch_sync, bucket, key)
    except S3Error as e:
        logger.error(f"[Proxy] MinIO 조회 실패: {bucket}/{key} → {e}")
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")

    content_type = _content_type_for(key)
    result = (image_bytes, content_type)

    # 캐시 저장 (단순 LRU 흉내: 상한 초과 시 가장 오래된 키 제거)
    if len(_image_cache) >= _IMAGE_CACHE_MAX:
        try:
            oldest_key = next(iter(_image_cache))
            del _image_cache[oldest_key]
        except StopIteration:
            pass
    _image_cache[(bucket, key)] = result

    return result


def _pick_from_library(library: dict[str, list[str]], category: str, used_paths: set[str]) -> str:
    candidates = [path for path in library[category] if path not in used_paths]
    if not candidates:
        candidates = library[category]
    selected = random.choice(candidates)
    used_paths.add(selected)
    return selected


async def _build_new_challenge_payload(
    request: Request,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], str | None]:
    """
    DB captcha_sets 우선 조회 → 없으면 MinIO fallback.
    모든 이미지 URL은 프록시 토큰으로 감싸서 반환.
    Returns: (emojis, photos, answer_indices, captcha_set_id)
    """
    # 1차: DB captcha_sets에서 조회
    db_set = await _fetch_captcha_set_from_db()
    if db_set and db_set["emojis"] and db_set["photos"]:
        logger.info(f"[Challenge] DB captcha_set 사용: {db_set['set_id']}")
        # raw key → 프록시 토큰 URL로 교체 (12개 redis SETEX를 병렬화)
        all_items = db_set["emojis"] + db_set["photos"]
        bucket_keys = [(item.pop("_bucket"), item["url"]) for item in all_items]
        tokens = await asyncio.gather(*[
            _create_image_token(bucket, key) for bucket, key in bucket_keys
        ])
        for item, token in zip(all_items, tokens):
            item["url"] = _build_proxy_url(token)
        return db_set["emojis"], db_set["photos"], db_set["answer_indices"], db_set["set_id"]

    # 2차: MinIO fallback (DB에 세트가 없을 때)
    logger.warning("[Challenge] DB captcha_set 없음 → MinIO fallback")
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
                f"캡챠용 동물 이미지가 부족합니다. "
                f"사용 가능: {available_categories} / 필요: 최소 3개 카테고리"
            ),
        )

    categories = random.sample(available_categories, 3)
    # 오답은 실사 사진 전체 카테고리에서 선택 (이모지 카테고리에 국한하지 않음)
    all_photo_categories = list(photo_lib.keys())
    wrong_categories = [c for c in all_photo_categories if c not in categories]
    if not wrong_categories:
        #  로컬처럼 사진 카테고리가 3종뿐인 경우에도 문제를 만들 수 있게 전체 사용 가능 카테고리에서 다시 고릅니다.
        wrong_categories = list(available_categories)
    answer_positions = sorted(random.sample(list(range(9)), 3))
    used_emoji_paths: set[str] = set()
    used_photo_paths: set[str] = set()

    emoji_bucket = settings.MINIO_EMOJI_BUCKET if EMOJI_ASSET_LIBRARY else settings.MINIO_PHOTO_BUCKET

    # 1단계: 모든 asset 경로를 먼저 결정 (CPU 작업, 빠름)
    emoji_specs: list[tuple[int, str, str]] = []  # (index, category, asset_path)
    for index, category in enumerate(categories):
        asset_path = _pick_from_library(emoji_lib, category, used_emoji_paths)
        emoji_specs.append((index, category, asset_path))

    photo_specs: list[tuple[int, str, str]] = []  # (position, category, asset_path)
    correct_index = 0
    for position in range(9):
        if position in answer_positions:
            category = categories[correct_index]
            correct_index += 1
        else:
            category = random.choice(wrong_categories)
        asset_path = _pick_from_library(photo_lib, category, used_photo_paths)
        photo_specs.append((position, category, asset_path))

    # 2단계: 12개 토큰을 병렬로 생성 (redis SETEX 1 RTT로 압축)
    token_coros = (
        [_create_image_token(emoji_bucket, spec[2]) for spec in emoji_specs] +
        [_create_image_token(settings.MINIO_PHOTO_BUCKET, spec[2]) for spec in photo_specs]
    )
    all_tokens = await asyncio.gather(*token_coros)
    emoji_tokens = all_tokens[:len(emoji_specs)]
    photo_tokens = all_tokens[len(emoji_specs):]

    emojis: list[dict[str, Any]] = [
        {
            "id": f"e-{index}",
            "url": _build_proxy_url(token),
            "category": category,
        }
        for (index, category, _), token in zip(emoji_specs, emoji_tokens)
    ]

    photos: list[dict[str, Any]] = [
        {
            "id": f"p-{position}",
            "url": _build_proxy_url(token),
            "index": position,
            "category": category,
        }
        for (position, category, _), token in zip(photo_specs, photo_tokens)
    ]

    return emojis, photos, answer_positions, None




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

    # 행동 벡터 먼저 생성 (pgvector KNN 레이어4에서 사용)
    behavior_vector = _build_behavior_vector(payload)
    fp_hash = _fingerprint_hash(payload)
    user_agent = request.headers.get("user-agent", "")

    (
        rule_score,
        vector_score,
        environment_score,
        header_score,
        fingerprint_score,
        final_score,
        immediate_block,
    ) = await _calculate_scores(payload, request, behavior_vector)

    if immediate_block:
        # DB: block 기록 (fire-and-forget, FK 순서 보장)
        block_session_id = str(uuid.uuid4())
        block_label, block_reason = _decide_label(
            outcome="init_block",
            rule_score=rule_score,
            fingerprint_score=fingerprint_score,
        )
        logger.info(f"[label.init_block] session={block_session_id} → {block_label} ({block_reason})")
        _bg(_save_session_then_embedding(
            session_id=block_session_id, trigger_type=payload.trigger_type,
            captcha_set_id=None, client_ip=client_ip, fingerprint_hash=fp_hash,
            behavior_score=rule_score, vector_score=vector_score,
            final_score=final_score, status_result="block",
            behavior_vector=behavior_vector, behavior_label=block_label,
        ))
        _bg(_save_bot_signature(
            client_ip, fp_hash, user_agent,
            "immediate_block_env_or_header", rule_score, final_score,
        ))
        return CaptchaInitResponse(
            status="block",
            message="비정상적인 브라우저 환경이 감지되었습니다.",
        )

    if final_score >= CAPTCHA_PASS_THRESHOLD:
        # PK 충돌 방지: 항상 새 UUID 생성 (다른 분기와 동일하게 통일)
        pass_session_id = str(uuid.uuid4())
        token = await _issue_captcha_token(client_ip, final_score, pass_session_id)
        pass_label, pass_reason = _decide_label(
            outcome="init_pass",
            rule_score=rule_score,
            fingerprint_score=fingerprint_score,
        )
        logger.info(f"[label.init_pass] session={pass_session_id} → {pass_label} ({pass_reason})")
        # DB: pass 기록 (fire-and-forget, FK 순서 보장)
        _bg(_save_session_then_embedding(
            session_id=pass_session_id, trigger_type=payload.trigger_type,
            captcha_set_id=None, client_ip=client_ip, fingerprint_hash=fp_hash,
            behavior_score=rule_score, vector_score=vector_score,
            final_score=final_score, status_result="pass",
            behavior_vector=behavior_vector, behavior_label=pass_label,
        ))
        return CaptchaInitResponse(status="pass", token=token)

    if final_score <= CAPTCHA_CHALLENGE_THRESHOLD:
        # DB: block (low score) 기록 (fire-and-forget, FK 순서 보장)
        block_session_id = str(uuid.uuid4())
        low_label, low_reason = _decide_label(
            outcome="init_block",
            rule_score=rule_score,
            fingerprint_score=fingerprint_score,
        )
        logger.info(f"[label.low_score_block] session={block_session_id} → {low_label} ({low_reason})")
        _bg(_save_session_then_embedding(
            session_id=block_session_id, trigger_type=payload.trigger_type,
            captcha_set_id=None, client_ip=client_ip, fingerprint_hash=fp_hash,
            behavior_score=rule_score, vector_score=vector_score,
            final_score=final_score, status_result="block",
            behavior_vector=behavior_vector, behavior_label=low_label,
        ))
        _bg(_save_bot_signature(
            client_ip, fp_hash, user_agent,
            "low_score_block", rule_score, final_score,
        ))
        return CaptchaInitResponse(
            status="block",
            message="보안 정책에 따라 접근이 제한되었습니다.",
        )

    # challenge 분기
    session_id = str(uuid.uuid4())
    session_payload = {
        "session_id": session_id,
        "client_ip": client_ip,
        "trigger_type": payload.trigger_type,
        "client_session_id": payload.session_id,
        "fingerprint_hash": fp_hash,
        "behavior_score": round(rule_score, 4),
        "vector_score": round(vector_score, 4),
        "environment_score": round(environment_score, 4),
        "header_score": round(header_score, 4),
        "fingerprint_score": round(fingerprint_score, 4),  # 라벨링 정책 평가에 필요
        "final_score": round(final_score, 4),
        "attempts": 0,
        "status": "challenge",
        "created_at": _now_time(),
    }
    await _save_json(_session_key(session_id), CAPTCHA_SESSION_TTL_SECONDS, session_payload)
    await _bind_active_session(client_ip, session_id)

    # DB: challenge 기록 (fire-and-forget, FK 순서 보장)
    _bg(_save_session_then_embedding(
        session_id=session_id, trigger_type=payload.trigger_type,
        captcha_set_id=None, client_ip=client_ip, fingerprint_hash=fp_hash,
        behavior_score=rule_score, vector_score=vector_score,
        final_score=final_score, status_result="challenge",
        behavior_vector=behavior_vector, behavior_label="unknown",
    ))

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
        emojis, photos, answer_indices, captcha_set_id = await _build_new_challenge_payload(request)
        session["emojis"] = emojis
        session["photos"] = photos
        session["answer_indices"] = answer_indices
        session["captcha_set_id"] = captcha_set_id
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

    solve_time_ms = int(solve_seconds * 1000)
    captcha_set_id = session.get("captcha_set_id")

    if is_correct:
        token = await _issue_captcha_token(
            client_ip,
            float(session.get("final_score", 0.5)),
            payload.session_id,
        )
        await redis_client.delete(_session_key(payload.session_id))
        await _clear_active_session(client_ip)

        # DB: verify 성공 업데이트
        # status='challenge_pass' 로 기록해 "비간섭 통과(pass)"와 "캡챠 풀고 통과(challenge_pass)"를 DB에서 구분.
        # 프론트 응답은 그대로 성공 토큰만 반환 (CaptchaVerifyResponse 는 status 문자열을 노출하지 않음).
        if _optional_table_enabled("captcha_sessions"):
            try:
                async with AsyncSessionLocal() as db:
                    await db.execute(text("""
                        UPDATE captcha_sessions
                        SET status = 'challenge_pass', attempt_count = :attempts,
                            solve_time_ms = :solve_ms, is_correct = true,
                            captcha_set_id = :captcha_set_id
                        WHERE id = :sid
                    """), {
                        "attempts": attempts + 1,
                        "solve_ms": solve_time_ms,
                        "captcha_set_id": captcha_set_id,
                        "sid": payload.session_id,
                    })
                    await db.commit()
            except Exception as e:
                if _is_missing_relation_error(e, "captcha_sessions"):
                    _disable_optional_table("captcha_sessions", e, "verify_challenge_pass")
                else:
                    logger.error(f"[DB] verify 성공 업데이트 실패: {e}")

        # behavior_embeddings.label 갱신 (까다로운 조건 통과 시에만 'human')
        verify_label, verify_reason = _decide_label(
            outcome="challenge_pass",
            rule_score=float(session.get("behavior_score", 0)),
            fingerprint_score=float(session.get("fingerprint_score", 0)),
            solve_time_ms=solve_time_ms,
        )
        logger.info(
            f"[label.challenge_pass] session={payload.session_id} → {verify_label} ({verify_reason})"
        )
        _bg(_update_embedding_label(payload.session_id, verify_label))

        return CaptchaVerifyResponse(success=True, token=token)

    attempts += 1
    remaining_attempts = max(CAPTCHA_MAX_ATTEMPTS - attempts, 0)

    if remaining_attempts == 0:
        session["attempts"] = attempts
        session["status"] = "blocked"
        await _save_json(_session_key(payload.session_id), CAPTCHA_SESSION_TTL_SECONDS, session)
        await _clear_active_session(client_ip)
        await _mark_lock(client_ip)

        # DB: 실패 초과 → block 업데이트 + bot_signatures
        if _optional_table_enabled("captcha_sessions"):
            try:
                async with AsyncSessionLocal() as db:
                    await db.execute(text("""
                        UPDATE captcha_sessions
                        SET status = 'block', attempt_count = :attempts,
                            solve_time_ms = :solve_ms, is_correct = false,
                            captcha_set_id = :captcha_set_id
                        WHERE id = :sid
                    """), {
                        "attempts": attempts,
                        "solve_ms": solve_time_ms,
                        "captcha_set_id": captcha_set_id,
                        "sid": payload.session_id,
                    })
                    await db.commit()
            except Exception as e:
                if _is_missing_relation_error(e, "captcha_sessions"):
                    _disable_optional_table("captcha_sessions", e, "verify_challenge_block")
                else:
                    logger.error(f"[DB] verify 실패초과 업데이트 실패: {e}")

        # behavior_embeddings.label 갱신 (까다로운 조건 통과 시에만 'bot')
        # 5회 모두 틀릴 때 평균 풀이 시간을 사용 (한 번이라도 정상 속도면 사람일 가능성)
        avg_solve_ms = solve_time_ms  # 마지막 시도 기준 (보수적 추정)
        fail_label, fail_reason = _decide_label(
            outcome="challenge_fail",
            rule_score=float(session.get("behavior_score", 0)),
            fingerprint_score=float(session.get("fingerprint_score", 0)),
            solve_time_ms=avg_solve_ms,
        )
        logger.info(
            f"[label.challenge_fail] session={payload.session_id} → {fail_label} ({fail_reason})"
        )
        _bg(_update_embedding_label(payload.session_id, fail_label))

        _bg(_save_bot_signature(
            client_ip,
            session.get("fingerprint_hash", ""),
            "",
            f"max_attempts_exceeded({attempts})",
            float(session.get("behavior_score", 0)),
            float(session.get("final_score", 0)),
        ))

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


# ─────────────────────────────────────────────
# DB: captcha_sets 기반 이미지 조회
# ─────────────────────────────────────────────


async def _fetch_captcha_set_from_db() -> dict[str, Any] | None:
    """DB captcha_sets에서 랜덤 1세트를 가져와 이미지 URL 구성"""
    if not _optional_table_enabled("captcha_sets"):
        return None
    try:
        async with AsyncSessionLocal() as db:
            # use_count 100회 미만에서 랜덤 선택
            result = await db.execute(text("""
                SELECT cs.id, cs.emoji_ids, cs.photo_ids, cs.answer_indices
                FROM captcha_sets cs
                WHERE cs.is_active = true AND cs.use_count < 100
                ORDER BY RANDOM()
                LIMIT 1
            """))
            row = result.fetchone()
            if not row:
                return None

            set_id, emoji_ids, photo_ids, answer_indices = row

            # use_count 증가
            await db.execute(text("""
                UPDATE captcha_sets SET use_count = use_count + 1 WHERE id = :set_id
            """), {"set_id": set_id})

            # emoji_images에서 category + image_key 조회
            emoji_result = await db.execute(text("""
                SELECT id, category, image_key FROM emoji_images
                WHERE id = ANY(:ids) AND is_active = true
            """), {"ids": emoji_ids})
            emoji_rows = emoji_result.fetchall()

            # real_photos에서 category + image_key 조회
            photo_result = await db.execute(text("""
                SELECT id, category, image_key FROM real_photos
                WHERE id = ANY(:ids) AND is_active = true
            """), {"ids": photo_ids})
            photo_rows = photo_result.fetchall()

            await db.commit()

            emojis = []
            emoji_map = {str(r[0]): (r[1], r[2]) for r in emoji_rows}
            for idx, eid in enumerate(emoji_ids):
                eid_str = str(eid)
                if eid_str in emoji_map:
                    cat, key = emoji_map[eid_str]
                    emojis.append({
                        "id": f"e-{idx}",
                        "url": key,           # 아직 raw key — _build_new_challenge_payload에서 토큰화
                        "category": cat,
                        "_bucket": settings.MINIO_EMOJI_BUCKET,
                    })

            photos = []
            photo_map = {str(r[0]): (r[1], r[2]) for r in photo_rows}
            for pos, pid in enumerate(photo_ids):
                pid_str = str(pid)
                if pid_str in photo_map:
                    cat, key = photo_map[pid_str]
                    photos.append({
                        "id": f"p-{pos}",
                        "url": key,           # raw key
                        "index": pos,
                        "category": cat,      # 서버 verify에서 selected_categories 비교 시 필요 (DB 캐시 경로 버그 수정)
                        "_bucket": settings.MINIO_PHOTO_BUCKET,
                    })

            return {
                "set_id": str(set_id),
                "emojis": emojis,
                "photos": photos,
                "answer_indices": list(answer_indices),
            }
    except Exception as e:
        if any(
            _is_missing_relation_error(e, table_name)
            for table_name in ("captcha_sets", "emoji_images", "real_photos")
        ):
            _disable_optional_table("captcha_sets", e, "_fetch_captcha_set_from_db")
            return None
        logger.error(f"[DB] captcha_sets 조회 실패: {e}")
        return None


# ─────────────────────────────────────────────
# DB: 백그라운드 fire-and-forget 헬퍼
# 메트릭/로깅 용 INSERT는 응답 경로에서 분리하여 latency 최소화
# ─────────────────────────────────────────────


def _bg(coro) -> None:
    """코루틴을 백그라운드 태스크로 던지고 즉시 반환.
    예외는 태스크 내부의 try/except에서 로그로 처리됨."""
    try:
        asyncio.create_task(coro)
    except RuntimeError:
        # 이벤트 루프가 없는 동기 컨텍스트에서 호출된 경우 무시
        pass


# ─────────────────────────────────────────────
# 라벨링 정책 (label noise 방지)
# ─────────────────────────────────────────────
# 배경: 캡챠 결과를 그대로 KNN 학습 라벨로 쓰면, 사람의 우연한 5회 실패나
# 봇의 우연한 통과로 학습 풀이 오염되어 시간이 지날수록 정확도가 떨어지는
# self-amplifying feedback loop 가 발생한다 (label noise).
#
# 해결책 (Day 1~2 Conservative Labeling Policy):
#  1) 옵션 A: 까다로운 조건을 모두 만족할 때만 human/bot 라벨 부여.
#             그 외는 'unknown' 으로 남겨 KNN 다수결에서 자연 배제.
#  2) 옵션 B (append-only): 한 번 부여된 비-unknown 라벨은 절대 덮어쓰지 않음.
#             잘못된 라벨이 후속 시도로 뒤집히는 일을 차단.

# human 라벨 조건 (전부 만족해야 함)
LABEL_HUMAN_RULE_MIN = 0.6        # 룰 점수 충분히 사람다움
LABEL_HUMAN_FP_MIN = 0.9          # 정상 브라우저 환경
LABEL_HUMAN_SOLVE_MIN_MS = 1500   # 최소 1.5초 (너무 빠르면 봇 의심)
LABEL_HUMAN_SOLVE_MAX_MS = 30000  # 최대 30초 (너무 느리면 라벨링 보류)

# bot 라벨 조건 (전부 만족해야 함)
LABEL_BOT_RULE_MAX = 0.4          # 룰 점수 충분히 봇다움
LABEL_BOT_FP_MAX = 0.7            # 환경도 의심
LABEL_BOT_SOLVE_MAX_MS = 1500     # 너무 빨리 풀었음

# init_block 전용 (즉시 차단된 환경 — 더 약한 조건으로 bot 부여 가능)
LABEL_INIT_BLOCK_RULE_MAX = 0.3
LABEL_INIT_BLOCK_FP_MAX = 0.7


def _decide_label(
    *,
    outcome: str,
    rule_score: float,
    fingerprint_score: float,
    solve_time_ms: int | None = None,
) -> tuple[str, str]:
    """outcome 별로 까다로운 조건을 만족할 때만 human/bot 라벨 부여.

    Args:
        outcome: 'init_pass' | 'init_block' | 'challenge_pass' | 'challenge_fail'
        rule_score: 레이어3 룰 기반 점수 (0.0 ~ 1.0)
        fingerprint_score: 핑거프린트 점수 (0.0 ~ 1.0)
        solve_time_ms: 그리드 풀이 시간 (init 분기는 None)

    Returns:
        (label, reason) — label 은 'human'/'bot'/'unknown', reason 은 결정 근거 로그용
    """
    if outcome in ("init_pass", "challenge_pass"):
        if rule_score < LABEL_HUMAN_RULE_MIN:
            return "unknown", f"rule={rule_score:.2f}<{LABEL_HUMAN_RULE_MIN}"
        if fingerprint_score < LABEL_HUMAN_FP_MIN:
            return "unknown", f"fp={fingerprint_score:.2f}<{LABEL_HUMAN_FP_MIN}"
        if outcome == "challenge_pass":
            if solve_time_ms is None:
                return "unknown", "solve_ms=None"
            if not (LABEL_HUMAN_SOLVE_MIN_MS <= solve_time_ms <= LABEL_HUMAN_SOLVE_MAX_MS):
                return "unknown", f"solve_ms={solve_time_ms} out of [{LABEL_HUMAN_SOLVE_MIN_MS},{LABEL_HUMAN_SOLVE_MAX_MS}]"
        return "human", "all conditions met"

    if outcome == "challenge_fail":
        if rule_score >= LABEL_BOT_RULE_MAX:
            return "unknown", f"rule={rule_score:.2f}>={LABEL_BOT_RULE_MAX}"
        if fingerprint_score >= LABEL_BOT_FP_MAX:
            return "unknown", f"fp={fingerprint_score:.2f}>={LABEL_BOT_FP_MAX}"
        if solve_time_ms is None or solve_time_ms >= LABEL_BOT_SOLVE_MAX_MS:
            return "unknown", f"solve_ms={solve_time_ms}>={LABEL_BOT_SOLVE_MAX_MS}"
        return "bot", "all conditions met"

    if outcome == "init_block":
        if rule_score < LABEL_INIT_BLOCK_RULE_MAX and fingerprint_score < LABEL_INIT_BLOCK_FP_MAX:
            return "bot", "init_block strong signal"
        return "unknown", f"init_block rule={rule_score:.2f} fp={fingerprint_score:.2f}"

    return "unknown", f"unhandled outcome={outcome}"


async def _update_embedding_label(session_id: str, label: str) -> None:
    """verify 결과에 따라 behavior_embeddings.label 갱신 (append-only).

    중요: 기존 라벨이 'unknown' 인 경우에만 UPDATE 한다.
    한 번 'human' 또는 'bot' 으로 확정된 라벨은 후속 호출로 절대 덮어쓰지 않는다.
    이는 잘못된 라벨이 향후 시도에 의해 뒤집히면서 학습 풀을 오염시키는 것을
    방지하기 위한 옵션 B (append-only) 정책의 일부.
    """
    if label == "unknown":
        # unknown 으로 되돌리는 일은 없음
        logger.info(f"[label.skip] session={session_id} → unknown 유지")
        return
    if not _optional_table_enabled("behavior_embeddings"):
        return
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(text("""
                UPDATE behavior_embeddings
                SET label = :label
                WHERE session_id = :sid AND label = 'unknown'
            """), {"label": label, "sid": session_id})
            await db.commit()
            if result.rowcount and result.rowcount > 0:
                logger.info(f"[label.assign] session={session_id} → {label}")
            else:
                logger.info(f"[label.skip] session={session_id} → 이미 라벨 확정됨, {label} 적용 안 함")
    except Exception as e:
        if _is_missing_relation_error(e, "behavior_embeddings"):
            _disable_optional_table("behavior_embeddings", e, "_update_embedding_label")
            return
        logger.error(f"[DB] behavior_embedding label 업데이트 실패: {e}")


async def _save_session_then_embedding(
    *,
    session_id: str,
    trigger_type: str,
    captcha_set_id: str | None,
    client_ip: str,
    fingerprint_hash: str,
    behavior_score: float,   # 레이어3 룰 점수
    vector_score: float,     # 레이어4 KNN 점수
    final_score: float,
    status_result: str,
    behavior_vector: list[float],
    behavior_label: str,
) -> None:
    """captcha_sessions INSERT가 commit된 후에 behavior_embeddings INSERT.
    behavior_embeddings.session_id 가 captcha_sessions.id 를 FK 참조하므로
    순서를 강제해야 ForeignKeyViolationError 방지.
    """
    await _save_captcha_session_to_db(
        session_id=session_id,
        trigger_type=trigger_type,
        captcha_set_id=captcha_set_id,
        client_ip=client_ip,
        fingerprint_hash=fingerprint_hash,
        behavior_score=behavior_score,
        vector_score=vector_score,
        final_score=final_score,
        status_result=status_result,
    )
    await _save_behavior_embedding(session_id, behavior_vector, behavior_label)


# ─────────────────────────────────────────────
# DB: captcha_sessions 저장 (모든 결과 기록)
# ─────────────────────────────────────────────


async def _save_captcha_session_to_db(
    session_id: str,
    trigger_type: str,
    captcha_set_id: str | None,
    client_ip: str,
    fingerprint_hash: str,
    behavior_score: float,       # 레이어3 룰 점수
    vector_score: float,         # 레이어4 KNN 점수
    final_score: float,
    status_result: str,          # pass / challenge / block
    attempt_count: int = 0,
    solve_time_ms: int | None = None,
    is_correct: bool | None = None,
) -> None:
    """captcha_sessions 테이블에 INSERT (발표 메트릭용)"""
    if not _optional_table_enabled("captcha_sessions"):
        return
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                INSERT INTO captcha_sessions (
                    id, trigger_type, captcha_set_id, client_ip,
                    behavior_score, vector_score,
                    final_score, status, attempt_count, solve_time_ms,
                    is_correct
                ) VALUES (
                    :id, :trigger_type, :captcha_set_id, CAST(:client_ip AS inet),
                    :behavior_score, :vector_score,
                    :final_score, :status, :attempt_count, :solve_time_ms,
                    :is_correct
                )
            """), {
                "id": session_id,
                "trigger_type": trigger_type,
                "captcha_set_id": captcha_set_id,
                "client_ip": client_ip if client_ip != "unknown" else "0.0.0.0",
                "behavior_score": round(behavior_score, 4),
                "vector_score": round(vector_score, 4),  # ← 버그 수정: 실제 KNN 점수 저장
                "final_score": round(final_score, 4),
                "status": status_result,
                "attempt_count": attempt_count,
                "solve_time_ms": solve_time_ms,
                "is_correct": is_correct,
            })
            await db.commit()
            logger.info(f"[DB] captcha_session 저장: {session_id} → {status_result}")
    except Exception as e:
        if _is_missing_relation_error(e, "captcha_sessions"):
            _disable_optional_table("captcha_sessions", e, "_save_captcha_session_to_db")
            return
        logger.error(f"[DB] captcha_session 저장 실패: {e}")


# ─────────────────────────────────────────────
# DB: behavior 벡터화 + pgvector 유사도 검색
# ─────────────────────────────────────────────


def _build_behavior_vector(payload: CaptchaInitRequest) -> list[float]:
    """
    행동 데이터를 15차원 벡터로 변환 (pgvector 호환)

    벡터 구성:
    [0]  speed_mean              마우스 속도 평균 — 봇은 일정한 속도
    [1]  speed_variance          마우스 속도 분산 — 봇은 분산이 극히 낮음
    [2]  direction_changes       방향 변화 횟수 — 봇은 직선 이동
    [3]  directness              직선 비율 — 봇은 1.0에 가까움
    [4]  total_distance          총 이동 거리 — 사람은 더 많이 돌아다님
    [5]  pause_count             정지 횟수(200ms+) — 사람은 멈추고 생각, 봇은 안 멈춤
    [6]  backtrack_count         되돌아감 횟수 — 사람은 실수로 되돌아감, 봇은 안 함
    [7]  click_spread_x          클릭 X 분산 — 봇은 정확한 좌표만 클릭
    [8]  click_spread_y          클릭 Y 분산 — 위와 동일
    [9]  click_interval_variance 클릭 간격 분산 — 봇은 일정한 간격
    [10] key_interval_variance   키 입력 간격 분산 — 봇은 일정한 타이핑
    [11] first_action_delay      첫 행동 지연(ms) — 봇은 즉시 행동
    [12] page_load_to_checkbox   체크박스 도달 시간(ms) — 봇은 극단적으로 빠름
    [13] scrolled                스크롤 여부 — 사람은 스크롤 자주 함
    [14] move_to_click_ratio     이동/클릭 비율 — 사람은 이동이 많고 클릭이 적음
    """
    moves = payload.mouse_moves
    clicks = payload.clicks

    # ── mouse features ──
    total_distance = 0.0
    speeds: list[float] = []
    direction_changes = 0
    pause_count = 0
    backtrack_count = 0
    prev_angle: float | None = None

    for prev_m, cur_m in zip(moves, moves[1:]):
        dx = cur_m.x - prev_m.x
        dy = cur_m.y - prev_m.y
        dt = max(cur_m.t - prev_m.t, 1)
        dist = math.hypot(dx, dy)
        total_distance += dist
        speeds.append(dist / dt)

        # 정지 감지: 200ms 이상 같은 위치 (이동거리 2px 이하)
        if dt >= 200 and dist <= 2.0:
            pause_count += 1

        # 방향 변화 + 되돌아감 감지
        angle = math.atan2(dy, dx)
        if prev_angle is not None:
            angle_diff = abs(angle - prev_angle)
            # 방향 변화: 약 31도 이상 꺾임
            if angle_diff > 0.55:
                direction_changes += 1
            # 되돌아감: 약 140도 이상 반전 (거의 반대 방향)
            if angle_diff > 2.44:
                backtrack_count += 1
        prev_angle = angle

    speed_mean = sum(speeds) / max(len(speeds), 1)
    speed_var = _variance(speeds)
    straight = math.hypot(
        moves[-1].x - moves[0].x, moves[-1].y - moves[0].y
    ) if len(moves) >= 2 else 0.0
    directness = straight / max(total_distance, 1.0)

    # ── click features ──
    click_count = len(clicks)
    cx = [c.x for c in clicks]
    cy = [c.y for c in clicks]
    spread_x = math.sqrt(_variance(cx)) if cx else 0.0
    spread_y = math.sqrt(_variance(cy)) if cy else 0.0
    click_intervals = [
        max(cur_c.t - prev_c.t, 1) for prev_c, cur_c in zip(clicks, clicks[1:])
    ]
    ci_var = _variance([float(v) for v in click_intervals])

    # ── timing features ──
    ki_var = _variance([float(v) for v in payload.key_intervals]) if payload.key_intervals else 0.0
    event_times = [m.t for m in moves] + [c.t for c in clicks]
    first_action = min(event_times) if event_times else payload.page_load_to_checkbox

    return [
        speed_mean,                                        # [0]
        speed_var,                                         # [1]
        float(direction_changes),                          # [2]
        directness,                                        # [3]
        total_distance,                                    # [4]
        float(pause_count),                                # [5]
        float(backtrack_count),                             # [6]
        spread_x,                                          # [7]
        spread_y,                                          # [8]
        ci_var,                                            # [9]
        ki_var,                                            # [10]
        float(first_action),                               # [11]
        float(payload.page_load_to_checkbox),              # [12]
        1.0 if payload.scrolled else 0.0,                  # [13]
        float(len(moves)) / max(click_count, 1),           # [14]
    ]


async def _save_behavior_embedding(
    session_id: str,
    vector: list[float],
    label: str,
) -> None:
    """behavior_embeddings 테이블에 INSERT"""
    if not _optional_table_enabled("behavior_embeddings"):
        return
    try:
        async with AsyncSessionLocal() as db:
            vec_str = "[" + ",".join(str(v) for v in vector) + "]"
            await db.execute(text("""
                INSERT INTO behavior_embeddings (id, session_id, vector, label)
                VALUES (:id, :session_id, CAST(:vector AS vector), :label)
            """), {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "vector": vec_str,
                "label": label,
            })
            await db.commit()
    except Exception as e:
        if _is_missing_relation_error(e, "behavior_embeddings"):
            _disable_optional_table("behavior_embeddings", e, "_save_behavior_embedding")
            return
        logger.error(f"[DB] behavior_embedding 저장 실패: {e}")


async def _search_similar_behaviors(vector: list[float], top_k: int = 5) -> list[dict]:
    """pgvector 코사인 유사도 검색으로 유사 행동 패턴 조회"""
    if not _optional_table_enabled("behavior_embeddings"):
        return []
    try:
        async with AsyncSessionLocal() as db:
            vec_str = "[" + ",".join(str(v) for v in vector) + "]"
            result = await db.execute(text("""
                SELECT label, 1 - (vector <=> CAST(:vec AS vector)) as similarity
                FROM behavior_embeddings
                ORDER BY vector <=> CAST(:vec AS vector)
                LIMIT :k
            """), {"vec": vec_str, "k": top_k})
            rows = result.fetchall()
            return [{"label": r[0], "similarity": round(r[1], 4)} for r in rows]
    except Exception as e:
        if _is_missing_relation_error(e, "behavior_embeddings"):
            _disable_optional_table("behavior_embeddings", e, "_search_similar_behaviors")
            return []
        logger.error(f"[DB] pgvector 검색 실패: {e}")
        return []


async def _calculate_vector_score(similar_results: list[dict]) -> float:
    """유사 행동 검색 결과로 봇 의심 점수 계산 (높을수록 사람).

    Cold Start: behavior_embeddings 가 비어있는 초반엔 0.7(인간 편향)을 반환.
    실제 가중치는 _calculate_scores 에서 sample_size 기준으로 조정되므로,
    여기서는 "데이터가 없을 때 정상 유저가 막히지 않도록" 하는 안전장치만 담당.
    """
    if not similar_results:
        return 0.7  # Cold Start: 인간 쪽 약한 편향

    # unknown 라벨(=challenge 중 미판정)은 집계에서 제외
    labeled = [r for r in similar_results if r.get("label") in ("human", "bot")]
    if not labeled:
        return 0.7  # 라벨이 전부 unknown → 초기 데이터 풀과 동일하게 취급

    bot_count = sum(1 for r in labeled if r["label"] == "bot")
    human_count = sum(1 for r in labeled if r["label"] == "human")
    total = len(labeled)

    human_ratio = human_count / total
    avg_sim = sum(r["similarity"] for r in labeled) / total

    # human 비율이 높고 유사도 높으면 → 높은 점수
    return _clamp(human_ratio * 0.7 + avg_sim * 0.3)


# ─────────────────────────────────────────────
# DB: bot_signatures INSERT (block 시 영구 기록)
# ─────────────────────────────────────────────


async def _save_bot_signature(
    client_ip: str,
    fingerprint_hash: str,
    user_agent: str,
    reason: str,
    behavior_score: float,
    final_score: float,
) -> None:
    """bot_signatures 테이블에 INSERT"""
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("""
                INSERT INTO bot_signatures (
                    id, ip_address, fingerprint_hash, reason
                ) VALUES (
                    :id, CAST(:ip AS inet), :fp, :reason
                )
            """), {
                "id": str(uuid.uuid4()),
                "ip": client_ip if client_ip != "unknown" else "0.0.0.0",
                "fp": fingerprint_hash,
                "reason": f"{reason} (b={behavior_score:.2f},f={final_score:.2f},ua={user_agent[:50]})",
            })
            await db.commit()
            logger.info(f"[DB] bot_signature 저장: {client_ip} / {reason}")
    except Exception as e:
        logger.error(f"[DB] bot_signature 저장 실패: {e}")
