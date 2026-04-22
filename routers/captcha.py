from fastapi import APIRouter, File, Form, UploadFile, Request
import httpx
import uuid
import random
import json
import traceback
import logging
import asyncio
import hashlib
from datetime import datetime
from io import BytesIO
from typing import Any, Optional

import redis.asyncio as redis
import asyncpg
from PIL import Image
from minio import Minio

from core.config import settings

router = APIRouter()
logger = logging.getLogger("handocr-api")


# ─────────────────────────────────────────────
# HandOCR 캡챠 설정
# ─────────────────────────────────────────────

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
GPU_SERVER_URL = settings.GPU_SERVER_URL

ALL_POSES = [
    "주먹 ✊",
    "손바닥 🖐️",
    "브이 ✌️",
    "따봉 👍",
]

MAX_ATTEMPTS = 5

DATABASE_URL = settings.DATABASE_URL
_db_pool: Optional[asyncpg.Pool] = None

minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
)

PHOTO_BUCKET = settings.MINIO_PHOTO_BUCKET
_photo_bucket_ready = False


# ─────────────────────────────────────────────
# IP / Rate Limit / Block 정책
# ─────────────────────────────────────────────

CAPTCHA_SESSION_TTL = 300
PASS_TOKEN_TTL = 180

# start 요청 제한
START_LIMIT_PER_MINUTE = 10
START_LIMIT_PER_10_MINUTES = 30

# verify 실패 누적 제한
VERIFY_FAIL_LIMIT_10M = 10
VERIFY_FAIL_LIMIT_1H = 30

# 차단 시간
BLOCK_TTL_SHORT = 10 * 60      # 10분
BLOCK_TTL_LONG = 60 * 60       # 1시간

# 활성 세션 1개 제한
ACTIVE_SESSION_TTL = CAPTCHA_SESSION_TTL


def normalize_asyncpg_dsn(database_url: str) -> str:
    if database_url.startswith("postgresql+asyncpg://"):
        return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return database_url


async def get_db_pool() -> asyncpg.Pool:
    global _db_pool
    if _db_pool is None:
        dsn = normalize_asyncpg_dsn(DATABASE_URL)
        _db_pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
    return _db_pool


def build_ai_failure_message(gpu_result: dict, remaining_attempts: int) -> str:
    error_code = gpu_result.get("error_code", "UNKNOWN_ERROR")
    detail = gpu_result.get("detail", "")
    guide = gpu_result.get("guide", "")

    title_map = {
        "HAND_NOT_DETECTED": "AI 검사에 실패했습니다. 사진에서 손을 찾지 못했어요.",
        "MULTIPLE_HANDS_DETECTED": "AI 검사에 실패했습니다. 사진에 손이 여러 개 보입니다.",
        "LOW_CONFIDENCE": "AI 검사에 실패했습니다. 손 모양을 확실하게 구분하지 못했어요.",
        "IMAGE_TOO_SMALL": "AI 검사에 실패했습니다. 사진 해상도가 너무 낮아요.",
        "IMAGE_DECODE_FAILED": "AI 검사에 실패했습니다. 이미지를 읽을 수 없어요.",
        "UNSUPPORTED_POSE": "AI 검사에 실패했습니다. 지원하지 않는 손 포즈로 인식됐어요.",
        "TEXT_NOT_DETECTED": "AI 검사에 실패했습니다. 5자리 문자+숫자를 찾지 못했어요.",
        "TEXT_LENGTH_INVALID": "AI 검사에 실패했습니다. 5자리 문자열이 선명하게 인식되지 않았어요.",
        "OCR_FAILED": "AI 검사에 실패했습니다. 문자 인식 중 오류가 발생했어요.",
        "HAND_LANDMARKER_FAILED": "AI 검사에 실패했습니다. 손 인식 모델 처리 중 오류가 발생했어요.",
        "MODEL_PREDICTION_FAILED": "AI 검사에 실패했습니다. 손 포즈 판별 중 오류가 발생했어요.",
        "EMPTY_IMAGE": "AI 검사에 실패했습니다. 업로드된 이미지가 비어 있어요.",
        "HAND_TOO_SMALL": "AI 검사에 실패했습니다. 손이 너무 작게 찍혔어요.",
    }

    lines = [title_map.get(error_code, f"AI 검사에 실패했습니다. {gpu_result.get('message', '')}")]

    if detail:
        lines.append(f"상세 사유: {detail}")

    if guide:
        lines.append(f"다시 시도하는 방법: {guide}")

    raw_candidates = gpu_result.get("ocr_text_candidates")
    if raw_candidates:
        lines.append(f"OCR 후보: {', '.join(raw_candidates[:5])}")

    lines.append("요구되는 손 포즈와 5자리 문자+숫자가 한 장의 사진 안에 선명하게 보여야 합니다.")
    lines.append(f"남은 기회: {remaining_attempts}회")
    return "\n".join(lines)


def safe_float(value: Any):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any):
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


CONFUSION_MAP = {
    "0": ["O"],
    "O": ["0"],
    "1": ["I"],
    "I": ["1"],
    "5": ["S"],
    "S": ["5"],
    "8": ["B"],
    "B": ["8"],
    "2": ["Z"],
    "Z": ["2"],
}


def generate_confusion_variants(text: str, max_variants: int = 50) -> list[str]:
    text = (text or "").strip().upper()
    if not text:
        return []

    variants = {text}

    for i, ch in enumerate(text):
        if ch in CONFUSION_MAP:
            new_variants = set()
            for base in variants:
                for repl in CONFUSION_MAP[ch]:
                    new_variants.add(base[:i] + repl + base[i + 1 :])
            variants |= new_variants
            if len(variants) >= max_variants:
                break

    return list(variants)[:max_variants]


def resolve_text_match_with_confusions(
    expected_text: str,
    detected_text: Optional[str],
    ocr_text_candidates: Optional[list],
) -> tuple[bool, str, Optional[str]]:
    expected = (expected_text or "").strip().upper()
    detected = (detected_text or "").strip().upper()

    if detected and detected == expected:
        return True, "exact", detected

    if detected:
        variants = generate_confusion_variants(detected)
        if expected in variants:
            return True, "confusion_detected_text", detected

    for candidate in (ocr_text_candidates or []):
        cand = str(candidate).strip().upper()
        if not cand:
            continue

        if cand == expected:
            return True, "candidate_exact", cand

        variants = generate_confusion_variants(cand)
        if expected in variants:
            return True, "confusion_candidate", cand

    return False, "no_match", detected or None


def guess_image_ext(upload: UploadFile) -> str:
    filename = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()

    if filename.endswith(".png") or content_type == "image/png":
        return ".png"
    if filename.endswith(".webp") or content_type == "image/webp":
        return ".webp"
    return ".jpg"


def guess_content_type(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").strip().lower()
    if content_type:
        return content_type

    ext = guess_image_ext(upload)
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def extract_image_size(image_bytes: bytes) -> tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return img.width, img.height
    except Exception:
        return None, None


def ensure_photo_bucket() -> None:
    global _photo_bucket_ready

    if _photo_bucket_ready:
        return

    if not minio_client.bucket_exists(PHOTO_BUCKET):
        minio_client.make_bucket(PHOTO_BUCKET)

    _photo_bucket_ready = True


def upload_original_image_to_minio(upload: UploadFile, image_bytes: bytes) -> dict:
    ensure_photo_bucket()

    sha256 = hashlib.sha256(image_bytes).hexdigest()
    width, height = extract_image_size(image_bytes)
    ext = guess_image_ext(upload)
    content_type = guess_content_type(upload)

    now = datetime.utcnow()
    object_key = (
        f"handocr/original/"
        f"{now:%Y/%m/%d}/"
        f"{sha256[:2]}/"
        f"{uuid.uuid4().hex}{ext}"
    )

    minio_client.put_object(
        PHOTO_BUCKET,
        object_key,
        BytesIO(image_bytes),
        length=len(image_bytes),
        content_type=content_type,
    )

    return {
        "image_key": object_key,
        "image_sha256": sha256,
        "image_width": width,
        "image_height": height,
    }


def normalize_text_region_bbox(text_region_bbox: Optional[dict], image_width: int, image_height: int) -> Optional[dict]:
    if not text_region_bbox:
        return None

    try:
        x_min = max(int(text_region_bbox.get("x_min", 0)), 0)
        y_min = max(int(text_region_bbox.get("y_min", 0)), 0)
        x_max = min(int(text_region_bbox.get("x_max", 0)), image_width)
        y_max = min(int(text_region_bbox.get("y_max", 0)), image_height)

        if x_max <= x_min or y_max <= y_min:
            return None

        normalized = dict(text_region_bbox)
        normalized["x_min"] = x_min
        normalized["y_min"] = y_min
        normalized["x_max"] = x_max
        normalized["y_max"] = y_max
        normalized["width"] = x_max - x_min
        normalized["height"] = y_max - y_min
        return normalized
    except Exception:
        return None


def upload_text_crop_to_minio(image_bytes: bytes, text_region_bbox: dict) -> tuple[Optional[str], Optional[dict]]:
    ensure_photo_bucket()

    with Image.open(BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")
        image_width, image_height = rgb.size

        normalized_bbox = normalize_text_region_bbox(text_region_bbox, image_width, image_height)
        if not normalized_bbox:
            return None, None

        crop = rgb.crop((
            normalized_bbox["x_min"],
            normalized_bbox["y_min"],
            normalized_bbox["x_max"],
            normalized_bbox["y_max"],
        ))

        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        crop_bytes = buffer.getvalue()

    crop_sha256 = hashlib.sha256(crop_bytes).hexdigest()
    now = datetime.utcnow()
    object_key = (
        f"handocr/text-crop/"
        f"{now:%Y/%m/%d}/"
        f"{crop_sha256[:2]}/"
        f"{uuid.uuid4().hex}.png"
    )

    minio_client.put_object(
        PHOTO_BUCKET,
        object_key,
        BytesIO(crop_bytes),
        length=len(crop_bytes),
        content_type="image/png",
    )

    return object_key, normalized_bbox


# ─────────────────────────────────────────────
# IP 관련 유틸
# ─────────────────────────────────────────────

def get_client_ip(request: Request) -> str:
    """
    운영에서 프록시(Nginx/ALB/Cloudflare) 뒤에 있다면
    해당 프록시를 신뢰하는 구성일 때만 X-Forwarded-For를 사용하세요.
    """
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


async def get_block_ttl(ip: str) -> int:
    ttl = await redis_client.ttl(f"captcha:block:{ip}")
    return ttl if ttl and ttl > 0 else 0


async def is_ip_blocked(ip: str) -> tuple[bool, int]:
    ttl = await get_block_ttl(ip)
    return ttl > 0, ttl


async def block_ip(ip: str, ttl_seconds: int, reason: str):
    block_value = {
        "reason": reason,
        "blocked": True,
    }
    await redis_client.setex(
        f"captcha:block:{ip}",
        ttl_seconds,
        json.dumps(block_value, ensure_ascii=False),
    )


async def register_start_rate_limit(ip: str) -> tuple[bool, Optional[int], dict]:
    key_1m = f"captcha:start:{ip}:1m"
    key_10m = f"captcha:start:{ip}:10m"

    count_1m = await redis_client.incr(key_1m)
    if count_1m == 1:
        await redis_client.expire(key_1m, 60)

    count_10m = await redis_client.incr(key_10m)
    if count_10m == 1:
        await redis_client.expire(key_10m, 600)

    if count_1m > START_LIMIT_PER_MINUTE:
        await block_ip(ip, 5 * 60, "too_many_start_requests_1m")
        return True, 5 * 60, {
            "type": "START_RATE_LIMIT",
            "count1m": count_1m,
            "count10m": count_10m,
        }

    if count_10m > START_LIMIT_PER_10_MINUTES:
        await block_ip(ip, BLOCK_TTL_SHORT, "too_many_start_requests_10m")
        return True, BLOCK_TTL_SHORT, {
            "type": "START_RATE_LIMIT",
            "count1m": count_1m,
            "count10m": count_10m,
        }

    return False, None, {
        "count1m": count_1m,
        "count10m": count_10m,
    }


async def register_verify_failure(ip: str) -> tuple[bool, Optional[int], dict]:
    key_10m = f"captcha:verify_fail:{ip}:10m"
    key_1h = f"captcha:verify_fail:{ip}:1h"

    fail_10m = await redis_client.incr(key_10m)
    if fail_10m == 1:
        await redis_client.expire(key_10m, 600)

    fail_1h = await redis_client.incr(key_1h)
    if fail_1h == 1:
        await redis_client.expire(key_1h, 3600)

    if fail_10m >= VERIFY_FAIL_LIMIT_10M:
        await block_ip(ip, BLOCK_TTL_SHORT, "too_many_verify_failures_10m")
        return True, BLOCK_TTL_SHORT, {
            "type": "IP_BLOCKED",
            "fail10m": fail_10m,
            "fail1h": fail_1h,
            "reason": "too_many_verify_failures_10m",
        }

    if fail_1h >= VERIFY_FAIL_LIMIT_1H:
        await block_ip(ip, BLOCK_TTL_LONG, "too_many_verify_failures_1h")
        return True, BLOCK_TTL_LONG, {
            "type": "IP_BLOCKED",
            "fail10m": fail_10m,
            "fail1h": fail_1h,
            "reason": "too_many_verify_failures_1h",
        }

    return False, None, {
        "fail10m": fail_10m,
        "fail1h": fail_1h,
    }


async def relax_verify_failures_on_success(ip: str):
    await redis_client.delete(f"captcha:verify_fail:{ip}:10m")


def make_blocked_response(ttl: int, message: Optional[str] = None) -> dict:
    return {
        "success": False,
        "message": message or f"반복된 요청 또는 실패로 인해 임시 차단되었습니다. {ttl}초 후 다시 시도해주세요.",
        "failureReason": {
            "type": "IP_BLOCKED",
            "retryAfterSeconds": ttl,
        }
    }


async def handle_verify_failure_common(
    *,
    ip: str,
    session_id: str,
    session_data: dict,
    message: str,
    failure_reason: dict,
):
    session_data["attempts"] = session_data.get("attempts", 0) + 1
    await redis_client.setex(
        f"captcha:{session_id}",
        CAPTCHA_SESSION_TTL,
        json.dumps(session_data, ensure_ascii=False),
    )

    blocked, ttl, fail_meta = await register_verify_failure(ip)
    if blocked:
        return make_blocked_response(
            ttl=ttl or BLOCK_TTL_SHORT,
            message=f"반복된 실패로 인해 임시 차단되었습니다. {ttl}초 후 다시 시도해주세요.",
        )

    failure_reason = dict(failure_reason)
    failure_reason["ipFailCount"] = {
        "fail10m": fail_meta.get("fail10m"),
        "fail1h": fail_meta.get("fail1h"),
    }

    return {
        "success": False,
        "message": message,
        "failureReason": failure_reason,
    }


# ─────────────────────────────────────────────
# 학습 샘플 저장
# ─────────────────────────────────────────────

async def save_hand_pose_sample(
    *,
    session_id: str,
    expected_pose: str,
    expected_text: str,
    verify_success: bool,
    pose_match: Optional[bool],
    text_match: Optional[bool],
    gpu_result: Optional[dict],
    image_key: Optional[str] = None,
    image_sha256: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    text_region_bbox: Optional[dict] = None,
    text_crop_key: Optional[str] = None,
):
    pool = await get_db_pool()

    request_id = None
    detected_pose = None
    predicted_label = None
    pose_confidence = None
    hand_area_ratio = None
    hand_bbox = None
    pose_features = None
    detected_text = None
    ocr_confidence = None
    ocr_low_confidence = None
    ocr_best_attempt = None
    ocr_text_candidates = None
    ocr_debug_top = None
    inspection = None
    ai_success = None
    ai_error_code = None
    ai_message = None
    ai_detail = None
    ai_guide = None

    if isinstance(gpu_result, dict):
        request_id = gpu_result.get("request_id")
        detected_pose = gpu_result.get("detected_pose")
        predicted_label = safe_int(gpu_result.get("predicted_label"))
        pose_confidence = safe_float(gpu_result.get("pose_confidence"))
        hand_area_ratio = safe_float(gpu_result.get("hand_area_ratio"))
        hand_bbox = gpu_result.get("hand_bbox")
        pose_features = gpu_result.get("pose_features")
        detected_text = gpu_result.get("detected_text")
        ocr_confidence = safe_float(gpu_result.get("ocr_confidence"))
        ocr_low_confidence = gpu_result.get("ocr_low_confidence")
        ocr_best_attempt = gpu_result.get("ocr_best_attempt")
        ocr_text_candidates = gpu_result.get("ocr_text_candidates")
        ocr_debug_top = gpu_result.get("ocr_debug_top")
        inspection = gpu_result.get("inspection")
        ai_success = gpu_result.get("success")
        ai_error_code = gpu_result.get("error_code")
        ai_message = gpu_result.get("message")
        ai_detail = gpu_result.get("detail")
        ai_guide = gpu_result.get("guide")

    insert_sql = """
    INSERT INTO hand_pose_samples (
        session_id,
        request_id,
        expected_pose,
        expected_text,
        detected_pose,
        predicted_label,
        pose_confidence,
        hand_area_ratio,
        hand_bbox,
        feature_vector,
        detected_text,
        ocr_confidence,
        ocr_low_confidence,
        ocr_best_attempt,
        ocr_text_candidates,
        ocr_debug_top,
        inspection,
        ai_success,
        ai_error_code,
        ai_message,
        ai_detail,
        ai_guide,
        verify_success,
        pose_match,
        text_match,
        image_key,
        image_sha256,
        image_width,
        image_height,
        text_region_bbox,
        text_crop_key,
        created_at
    )
    VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8,
        $9::jsonb, $10::jsonb, $11, $12, $13, $14,
        $15::jsonb, $16::jsonb, $17::jsonb,
        $18, $19, $20, $21, $22, $23, $24, $25,
        $26, $27, $28, $29, $30::jsonb, $31, NOW()
    )
    """

    async with pool.acquire() as conn:
        await conn.execute(
            insert_sql,
            session_id,
            request_id,
            expected_pose,
            expected_text,
            detected_pose,
            predicted_label,
            pose_confidence,
            hand_area_ratio,
            json.dumps(hand_bbox, ensure_ascii=False) if hand_bbox is not None else None,
            json.dumps(pose_features, ensure_ascii=False) if pose_features is not None else None,
            detected_text,
            ocr_confidence,
            ocr_low_confidence,
            ocr_best_attempt,
            json.dumps(ocr_text_candidates, ensure_ascii=False) if ocr_text_candidates is not None else None,
            json.dumps(ocr_debug_top, ensure_ascii=False) if ocr_debug_top is not None else None,
            json.dumps(inspection, ensure_ascii=False) if inspection is not None else None,
            ai_success,
            ai_error_code,
            ai_message,
            ai_detail,
            ai_guide,
            verify_success,
            pose_match,
            text_match,
            image_key,
            image_sha256,
            image_width,
            image_height,
            json.dumps(text_region_bbox, ensure_ascii=False) if text_region_bbox is not None else None,
            text_crop_key,
        )


# ─────────────────────────────────────────────
# Start CAPTCHA
# ─────────────────────────────────────────────

@router.post("/captcha/handocr/start")
async def start_captcha(request: Request):
    ip = get_client_ip(request)

    blocked, ttl = await is_ip_blocked(ip)
    if blocked:
        return make_blocked_response(
            ttl,
            f"요청이 너무 많아 임시 차단되었습니다. {ttl}초 후 다시 시도해주세요."
        )

    limited, limit_ttl, _ = await register_start_rate_limit(ip)
    if limited:
        return make_blocked_response(
            limit_ttl or BLOCK_TTL_SHORT,
            f"문제 생성 요청이 너무 많습니다. {limit_ttl}초 후 다시 시도해주세요.",
        )

    active_session_key = f"captcha:active_session:{ip}"
    existing_session_id = await redis_client.get(active_session_key)

    if existing_session_id:
        await redis_client.delete(f"captcha:{existing_session_id}")
        await redis_client.delete(active_session_key)

    chars = "ABCDEFGHIJKLMNPQRSTUVWXYZ123456789"
    random_text = "".join(random.choice(chars) for _ in range(5))
    random_pose = random.choice(ALL_POSES)

    session_id = str(uuid.uuid4())
    session_data = {
        "text": random_text,
        "pose": random_pose,
        "attempts": 0,
        "ip": ip,
    }

    await redis_client.setex(
        f"captcha:{session_id}",
        CAPTCHA_SESSION_TTL,
        json.dumps(session_data, ensure_ascii=False),
    )
    await redis_client.setex(active_session_key, ACTIVE_SESSION_TTL, session_id)

    return {
        "success": True,
        "sessionId": session_id,
        "text": random_text,
        "pose": random_pose,
    }


# ─────────────────────────────────────────────
# Verify CAPTCHA
# ─────────────────────────────────────────────

@router.post("/captcha/handocr/verify")
async def verify_captcha(
    request: Request,
    sessionId: str = Form(...),
    image: UploadFile = File(...),
):
    ip = get_client_ip(request)

    blocked, ttl = await is_ip_blocked(ip)
    if blocked:
        return make_blocked_response(ttl)

    session_str = await redis_client.get(f"captcha:{sessionId}")
    if not session_str:
        return {
            "success": False,
            "message": "유효하지 않거나 5분이 지나 만료된 세션입니다. 새로고침 후 다시 시작해주세요.",
            "failureReason": {
                "type": "SESSION_EXPIRED",
            }
        }

    session_data = json.loads(session_str)
    expected_pose = session_data["pose"]
    expected_text = session_data["text"]
    session_ip = session_data.get("ip")

    if session_ip and session_ip != ip:
        return await handle_verify_failure_common(
            ip=ip,
            session_id=sessionId,
            session_data=session_data,
            message="세션을 발급받은 환경과 현재 요청 환경이 다릅니다. 다시 시작해주세요.",
            failure_reason={
                "type": "SESSION_IP_MISMATCH",
                "expectedSessionIp": session_ip,
            }
        )

    if session_data.get("attempts", 0) >= MAX_ATTEMPTS:
        await redis_client.delete(f"captcha:{sessionId}")
        if session_ip:
            await redis_client.delete(f"captcha:active_session:{session_ip}")
        return {
            "success": False,
            "message": "실패 횟수(5회)를 초과했습니다. 새로고침하여 처음부터 다시 시도해주세요.",
            "failureReason": {
                "type": "MAX_SESSION_ATTEMPTS_EXCEEDED",
            }
        }

    image_bytes = await image.read()
    if not image_bytes:
        return await handle_verify_failure_common(
            ip=ip,
            session_id=sessionId,
            session_data=session_data,
            message=(
                "업로드된 이미지가 비어 있습니다.\n"
                "사진을 다시 찍거나 다른 파일을 업로드해주세요.\n"
                f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
            ),
            failure_reason={
                "type": "EMPTY_IMAGE",
                "expectedPose": expected_pose,
                "expectedText": expected_text,
            }
        )

    image_meta = {
        "image_key": None,
        "image_sha256": None,
        "image_width": None,
        "image_height": None,
    }

    try:
        image_meta = await asyncio.to_thread(
            upload_original_image_to_minio,
            image,
            image_bytes,
        )
    except Exception:
        logger.exception("failed to upload original image to minio")

    gpu_result = None
    gpu_status_code = None
    gpu_raw_text = None
    text_region_bbox = None
    text_crop_key = None

    timeout = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        files = {
            "image": (
                image.filename or "upload.jpg",
                image_bytes,
                image.content_type or "application/octet-stream"
            )
        }

        try:
            response = await client.post(
                GPU_SERVER_URL,
                files=files
            )
            gpu_status_code = response.status_code
            gpu_raw_text = response.text[:2000]

            logger.info("GPU status=%s", response.status_code)
            logger.info("GPU content-type=%s", response.headers.get("content-type"))
            logger.info("GPU raw text=%s", gpu_raw_text)

            response.raise_for_status()

            try:
                gpu_result = response.json()

                if isinstance(gpu_result, dict):
                    text_region_bbox = gpu_result.get("text_region_bbox")

                    if text_region_bbox:
                        try:
                            text_crop_key, text_region_bbox = await asyncio.to_thread(
                                upload_text_crop_to_minio,
                                image_bytes,
                                text_region_bbox,
                            )
                        except Exception:
                            logger.exception("failed to upload text crop to minio")

            except Exception as json_error:
                logger.exception("GPU json parse failed")

                try:
                    await save_hand_pose_sample(
                        session_id=sessionId,
                        expected_pose=expected_pose,
                        expected_text=expected_text,
                        verify_success=False,
                        pose_match=None,
                        text_match=None,
                        gpu_result={
                            "success": False,
                            "error_code": "GPU_JSON_PARSE_FAILED",
                            "message": "AI 서버 응답을 해석하지 못했습니다.",
                            "detail": repr(json_error),
                        },
                        image_key=image_meta.get("image_key"),
                        image_sha256=image_meta.get("image_sha256"),
                        image_width=image_meta.get("image_width"),
                        image_height=image_meta.get("image_height"),
                        text_region_bbox=None,
                        text_crop_key=None,
                    )
                except Exception:
                    logger.exception("failed to save failed hand pose sample")

                return await handle_verify_failure_common(
                    ip=ip,
                    session_id=sessionId,
                    session_data=session_data,
                    message=(
                        "AI 서버 응답을 해석하지 못했습니다.\n"
                        "AI 서버가 JSON이 아닌 응답을 반환했습니다.\n"
                        f"상세: {repr(json_error)}\n"
                        f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
                    ),
                    failure_reason={
                        "type": "GPU_JSON_PARSE_FAILED",
                        "expectedPose": expected_pose,
                        "expectedText": expected_text,
                        "statusCode": gpu_status_code,
                        "rawText": gpu_raw_text,
                        "detail": repr(json_error),
                        "gpuServerUrl": GPU_SERVER_URL,
                    }
                )

            logger.info("GPU json=%s", gpu_result)

        except httpx.ConnectTimeout as e:
            logger.exception("GPU connect timeout")
            return await handle_verify_failure_common(
                ip=ip,
                session_id=sessionId,
                session_data=session_data,
                message=(
                    "AI 서버 연결 시간이 초과되었습니다.\n"
                    "GPU 서버가 응답 가능한 상태인지 확인해주세요.\n"
                    f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
                ),
                failure_reason={
                    "type": "GPU_CONNECT_TIMEOUT",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            )

        except httpx.ConnectError as e:
            logger.exception("GPU connect error")
            return await handle_verify_failure_common(
                ip=ip,
                session_id=sessionId,
                session_data=session_data,
                message=(
                    "AI 서버 연결에 실패했습니다.\n"
                    "GPU 서버가 꺼져 있거나, 주소/포트가 잘못되었거나, 네트워크에 문제가 있을 수 있습니다.\n"
                    f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
                ),
                failure_reason={
                    "type": "GPU_CONNECT_ERROR",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            )

        except httpx.ReadTimeout as e:
            logger.exception("GPU read timeout")
            return await handle_verify_failure_common(
                ip=ip,
                session_id=sessionId,
                session_data=session_data,
                message=(
                    "AI 서버가 사진을 분석하는 데 시간이 너무 오래 걸리고 있습니다.\n"
                    "서버는 살아 있지만 응답이 지연되고 있을 수 있습니다.\n"
                    f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
                ),
                failure_reason={
                    "type": "GPU_READ_TIMEOUT",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            )

        except httpx.HTTPStatusError as e:
            logger.exception("GPU HTTP status error")

            response_text = ""
            try:
                response_text = e.response.text[:2000]
            except Exception:
                response_text = ""

            status_code = e.response.status_code if e.response else None

            message = "AI 서버가 오류 응답을 반환했습니다."
            if status_code == 504:
                message = "AI 서버 응답이 지연되어 게이트웨이 타임아웃이 발생했습니다."
            elif status_code == 503:
                message = "AI 서버가 현재 요청을 처리할 수 없는 상태입니다."
            elif status_code == 502:
                message = "프록시 서버가 AI 서버와 정상 통신하지 못했습니다."
            elif status_code == 500:
                message = "AI 서버 내부 오류가 발생했습니다."

            return await handle_verify_failure_common(
                ip=ip,
                session_id=sessionId,
                session_data=session_data,
                message=(
                    f"{message}\n"
                    f"HTTP 상태 코드: {status_code}\n"
                    f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
                ),
                failure_reason={
                    "type": "GPU_HTTP_ERROR",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "statusCode": status_code,
                    "responseText": response_text,
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            )

        except httpx.RequestError as e:
            logger.exception("GPU request error")
            return await handle_verify_failure_common(
                ip=ip,
                session_id=sessionId,
                session_data=session_data,
                message=(
                    "AI 서버 요청 처리 중 네트워크 오류가 발생했습니다.\n"
                    f"상세: {repr(e)}\n"
                    f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
                ),
                failure_reason={
                    "type": "GPU_REQUEST_ERROR",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            )

        except Exception as e:
            logger.exception("GPU request/parse failed")
            traceback.print_exc()
            return await handle_verify_failure_common(
                ip=ip,
                session_id=sessionId,
                session_data=session_data,
                message=(
                    "AI 서버 통신 처리 중 알 수 없는 오류가 발생했습니다.\n"
                    f"상세: {repr(e)}\n"
                    f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
                ),
                failure_reason={
                    "type": "GPU_UNKNOWN_ERROR",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            )

    if not isinstance(gpu_result, dict):
        return await handle_verify_failure_common(
            ip=ip,
            session_id=sessionId,
            session_data=session_data,
            message=(
                "AI 서버 응답 형식이 올바르지 않습니다.\n"
                f"남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"
            ),
            failure_reason={
                "type": "GPU_INVALID_RESPONSE",
                "expectedPose": expected_pose,
                "expectedText": expected_text,
                "gpuResultType": str(type(gpu_result)),
                "gpuServerUrl": GPU_SERVER_URL,
            }
        )

    if not gpu_result.get("success"):
        try:
            await save_hand_pose_sample(
                session_id=sessionId,
                expected_pose=expected_pose,
                expected_text=expected_text,
                verify_success=False,
                pose_match=None,
                text_match=None,
                gpu_result=gpu_result,
                image_key=image_meta.get("image_key"),
                image_sha256=image_meta.get("image_sha256"),
                image_width=image_meta.get("image_width"),
                image_height=image_meta.get("image_height"),
                text_region_bbox=text_region_bbox,
                text_crop_key=text_crop_key,
            )
        except Exception:
            logger.exception("failed to save failed hand pose sample")

        return await handle_verify_failure_common(
            ip=ip,
            session_id=sessionId,
            session_data=session_data,
            message=build_ai_failure_message(
                gpu_result,
                MAX_ATTEMPTS - (session_data.get("attempts", 0) + 1),
            ),
            failure_reason={
                "type": "AI_DETECTION_FAILED",
                "expectedPose": expected_pose,
                "expectedText": expected_text,
                "aiErrorCode": gpu_result.get("error_code"),
                "aiMessage": gpu_result.get("message"),
                "aiDetail": gpu_result.get("detail"),
                "aiGuide": gpu_result.get("guide"),
                "ocrCandidates": gpu_result.get("ocr_text_candidates"),
                "inspection": gpu_result.get("inspection"),
                "gpuServerUrl": GPU_SERVER_URL,
            }
        )

    detected_pose = gpu_result.get("detected_pose")
    pose_confidence = safe_float(gpu_result.get("pose_confidence"))
    detected_text = gpu_result.get("detected_text")
    ocr_confidence = safe_float(gpu_result.get("ocr_confidence"))
    ocr_low_confidence = gpu_result.get("ocr_low_confidence", False)
    ocr_text_candidates = gpu_result.get("ocr_text_candidates", [])

    expected_text_normalized = (expected_text or "").strip().upper()
    detected_text_normalized = (detected_text or "").strip().upper()

    pose_ok = detected_pose == expected_pose
    raw_text_match = detected_text_normalized == expected_text_normalized

    text_ok, text_match_mode, matched_candidate = resolve_text_match_with_confusions(
        expected_text=expected_text,
        detected_text=detected_text,
        ocr_text_candidates=ocr_text_candidates,
    )

    if isinstance(gpu_result, dict):
        inspection = gpu_result.get("inspection") or {}
        inspection["api_raw_text_match"] = raw_text_match
        inspection["api_text_match_mode"] = text_match_mode
        inspection["api_matched_candidate"] = matched_candidate
        inspection["api_expected_text"] = expected_text
        inspection["api_detected_text"] = detected_text
        gpu_result["inspection"] = inspection

    try:
        await save_hand_pose_sample(
            session_id=sessionId,
            expected_pose=expected_pose,
            expected_text=expected_text,
            verify_success=(pose_ok and text_ok),
            pose_match=pose_ok,
            text_match=text_ok,
            gpu_result=gpu_result,
            image_key=image_meta.get("image_key"),
            image_sha256=image_meta.get("image_sha256"),
            image_width=image_meta.get("image_width"),
            image_height=image_meta.get("image_height"),
            text_region_bbox=text_region_bbox,
            text_crop_key=text_crop_key,
        )
    except Exception:
        logger.exception("failed to save hand pose sample")

    if not pose_ok or not text_ok:
        reasons = []

        if not pose_ok:
            reasons.append(f"손 포즈 불일치 (요구: {expected_pose} / 인식: {detected_pose})")

        if not text_ok:
            if ocr_low_confidence and ocr_confidence is not None:
                reasons.append(
                    f"문자 인식 신뢰도가 낮습니다. "
                    f"(요구: {expected_text} / 인식: {detected_text}, OCR 신뢰도: {ocr_confidence:.2f})"
                )
            else:
                reasons.append(
                    f"문자열 불일치 (요구: {expected_text} / 인식: {detected_text}, "
                    f"매칭 방식: {text_match_mode})"
                )

        message = "AI 검사에 실패했습니다.\n" + "\n".join(reasons)

        if pose_confidence is not None:
            message += f"\n손 포즈 신뢰도: {pose_confidence:.2f}"

        if ocr_confidence is not None:
            message += f"\nOCR 신뢰도: {ocr_confidence:.2f}"

        message += "\n손 포즈와 5자리 문자+숫자가 모두 선명하게 보이도록 다시 촬영해주세요."
        message += f"\n남은 기회: {MAX_ATTEMPTS - (session_data.get('attempts', 0) + 1)}회"

        return await handle_verify_failure_common(
            ip=ip,
            session_id=sessionId,
            session_data=session_data,
            message=message,
            failure_reason={
                "type": "MISSION_MISMATCH",
                "expectedPose": expected_pose,
                "detectedPose": detected_pose,
                "expectedText": expected_text,
                "detectedText": detected_text,
                "poseConfidence": pose_confidence,
                "ocrConfidence": ocr_confidence,
                "ocrCandidates": gpu_result.get("ocr_text_candidates"),
                "inspection": gpu_result.get("inspection"),
                "gpuServerUrl": GPU_SERVER_URL,
            }
        )

    pass_token = str(uuid.uuid4())
    await redis_client.setex(f"captcha_pass:{pass_token}", PASS_TOKEN_TTL, "PASSED")

    await redis_client.delete(f"captcha:{sessionId}")
    await redis_client.delete(f"captcha:active_session:{ip}")

    await relax_verify_failures_on_success(ip)

    return {
        "success": True,
        "message": "인증이 완료되었습니다.",
        "passToken": pass_token
    }