from __future__ import annotations

from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse
import json

import asyncpg
import httpx
import redis.asyncio as redis
<<<<<<< Updated upstream
from fastapi import APIRouter, Depends, HTTPException, Query
=======
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
>>>>>>> Stashed changes
from minio import Minio
from minio.error import S3Error

from core.config import settings
from routers.admin import require_admin_handocr_permission

router = APIRouter(
    prefix="/admin/handocr",
    tags=["Admin HandOCR"],
    dependencies=[Depends(require_admin_handocr_permission)],
)

DATABASE_URL = settings.DATABASE_URL
REDIS_URL = settings.REDIS_URL
GPU_SERVER_URL = settings.GPU_SERVER_URL

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

_db_pool: Optional[asyncpg.Pool] = None

minio_client = Minio(
    settings.MINIO_ENDPOINT,
    access_key=settings.MINIO_ACCESS_KEY,
    secret_key=settings.MINIO_SECRET_KEY,
    secure=settings.MINIO_SECURE,
)

PHOTO_BUCKET = settings.MINIO_PHOTO_BUCKET


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


def build_gpu_health_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    path = parsed.path or ""

    if path.endswith("/ai/predict/mission"):
        path = path[: -len("/ai/predict/mission")] + "/health"
    elif path.endswith("/ai/predict/pose"):
        path = path[: -len("/ai/predict/pose")] + "/health"
    else:
        path = "/health"

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            "",
            "",
            "",
        )
    )


def row_to_record(row: asyncpg.Record) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "request_id": row["request_id"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "verify_success": row["verify_success"],
        "expected_pose": row["expected_pose"],
        "detected_pose": row["detected_pose"],
        "expected_text": row["expected_text"],
        "detected_text": row["detected_text"],
        "pose_confidence": float(row["pose_confidence"]) if row["pose_confidence"] is not None else None,
        "ocr_confidence": float(row["ocr_confidence"]) if row["ocr_confidence"] is not None else None,
        "ocr_low_confidence": row["ocr_low_confidence"],
        "pose_match": row["pose_match"],
        "text_match": row["text_match"],
        "ai_error_code": row["ai_error_code"],
        "ai_message": row["ai_message"],
        "ai_guide": row["ai_guide"],
        "image_key": row["image_key"],
        "text_crop_key": row["text_crop_key"],
        "ocr_best_attempt": row["ocr_best_attempt"],
        "ocr_text_candidates": row["ocr_text_candidates"],
        "text_region_bbox": row["text_region_bbox"],
        "inspection": row["inspection"],
    }


def apply_status_tab_conditions(
    status_tab: Optional[str],
    conditions: list[str],
) -> None:
    if not status_tab or status_tab == "전체":
        return

    if status_tab == "성공":
        conditions.append("verify_success = TRUE")
    elif status_tab == "실패":
        conditions.append("verify_success = FALSE")
    elif status_tab == "저신뢰":
        conditions.append("ocr_low_confidence = TRUE")
    elif status_tab == "포즈불일치":
        conditions.append("pose_match = FALSE")
    elif status_tab == "문자불일치":
        conditions.append("text_match = FALSE")


async def collect_redis_keys(pattern: str, limit: int = 300) -> list[str]:
    keys: list[str] = []
    async for key in redis_client.scan_iter(match=pattern, count=200):
        keys.append(key)
        if len(keys) >= limit:
            break
    return keys


async def safe_get_ttl(key: str) -> int:
    ttl = await redis_client.ttl(key)
    return ttl if isinstance(ttl, int) and ttl > 0 else 0


def safe_json_loads(value: Optional[str]) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


@router.get("/records")
async def get_admin_handocr_records(
    keyword: Optional[str] = Query(default=None),
    date_from: Optional[date] = Query(default=None),
    date_to: Optional[date] = Query(default=None),
    error_code: Optional[str] = Query(default=None),
    pose: Optional[str] = Query(default=None),
    status_tab: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=300),
):
    pool = await get_db_pool()

    conditions: list[str] = []
    values: list[Any] = []

    if keyword:
        idx = len(values) + 1
        like = f"%{keyword}%"
        conditions.append(
            f"""(
                session_id::text ILIKE ${idx}
                OR COALESCE(request_id::text, '') ILIKE ${idx}
                OR expected_text ILIKE ${idx}
                OR COALESCE(detected_text, '') ILIKE ${idx}
            )"""
        )
        values.append(like)

    if date_from:
        idx = len(values) + 1
        conditions.append(f"created_at >= ${idx}")
        values.append(datetime.combine(date_from, datetime.min.time()))

    if date_to:
        idx = len(values) + 1
        conditions.append(f"created_at < ${idx}")
        values.append(datetime.combine(date_to + timedelta(days=1), datetime.min.time()))

    if error_code:
        idx = len(values) + 1
        conditions.append(f"ai_error_code = ${idx}")
        values.append(error_code)

    if pose:
        idx = len(values) + 1
        conditions.append(f"(expected_pose = ${idx} OR detected_pose = ${idx})")
        values.append(pose)

    apply_status_tab_conditions(status_tab, conditions)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    limit_idx = len(values) + 1
    values.append(limit)

    query = f"""
        SELECT
            session_id,
            request_id,
            created_at,
            verify_success,
            expected_pose,
            detected_pose,
            expected_text,
            detected_text,
            pose_confidence,
            ocr_confidence,
            ocr_low_confidence,
            pose_match,
            text_match,
            ai_error_code,
            ai_message,
            ai_guide,
            image_key,
            text_crop_key,
            ocr_best_attempt,
            ocr_text_candidates,
            text_region_bbox,
            inspection
        FROM hand_pose_samples
        {where_clause}
        ORDER BY created_at DESC
        LIMIT ${limit_idx}
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *values)

    return {"items": [row_to_record(row) for row in rows]}


@router.get("/health")
async def get_admin_handocr_health():
    health_url = build_gpu_health_url(GPU_SERVER_URL)
    timeout = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(health_url)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GPU health 조회 실패: {repr(e)}")


@router.get("/image")
async def get_admin_handocr_image(
    key: str = Query(..., min_length=1),
):
    try:
        obj = minio_client.get_object(PHOTO_BUCKET, key)
        try:
            data = obj.read()
            content_type = obj.headers.get(
                "Content-Type",
                "application/octet-stream",
            )
        finally:
            obj.close()
            obj.release_conn()

        return StreamingResponse(
            BytesIO(data),
            media_type=content_type,
            headers={
                "Cache-Control": "private, max-age=300",
                "Content-Disposition": f'inline; filename=\"{key.split("/")[-1]}\"',
            },
        )
    except S3Error as e:
        raise HTTPException(
            status_code=404,
            detail=f"이미지를 찾을 수 없습니다: {e.code}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"이미지 조회 실패: {repr(e)}",
        )


@router.get("/blocks")
async def get_admin_handocr_blocks(
    keyword: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
):
    keys = await collect_redis_keys("captcha:block:*", limit=limit)
    items: list[dict[str, Any]] = []

    for key in keys:
        ip = key.replace("captcha:block:", "", 1)
        value = await redis_client.get(key)
        ttl = await safe_get_ttl(key)
        payload = safe_json_loads(value)

        reason = payload.get("reason")
        blocked = bool(payload.get("blocked", True))

        if keyword:
            keyword_lower = keyword.lower()
            if keyword_lower not in ip.lower() and keyword_lower not in str(reason or "").lower():
                continue

        items.append(
            {
                "ip": ip,
                "blocked": blocked,
                "reason": reason,
                "ttl_seconds": ttl,
            }
        )

    items.sort(key=lambda x: x["ttl_seconds"], reverse=True)
    return {"items": items}


@router.post("/blocks/{ip}/release")
async def release_admin_handocr_block(ip: str):
    block_key = f"captcha:block:{ip}"
    deleted = await redis_client.delete(block_key)

    return {
        "success": True,
        "ip": ip,
        "released": deleted > 0,
    }


@router.post("/ips/{ip}/reset-failures")
async def reset_admin_handocr_ip_failures(ip: str):
    deleted = await redis_client.delete(
        f"captcha:verify_fail:{ip}:10m",
        f"captcha:verify_fail:{ip}:1h",
        f"captcha:start:{ip}:1m",
        f"captcha:start:{ip}:10m",
    )

    return {
        "success": True,
        "ip": ip,
        "deleted_key_count": deleted,
    }


@router.get("/sessions")
async def get_admin_handocr_sessions(
    keyword: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
):
    keys = await collect_redis_keys("captcha:active_session:*", limit=limit)
    items: list[dict[str, Any]] = []

    for active_key in keys:
        ip = active_key.replace("captcha:active_session:", "", 1)
        session_id = await redis_client.get(active_key)
        active_ttl = await safe_get_ttl(active_key)

        session_payload: dict[str, Any] = {}
        session_ttl = 0

        if session_id:
            session_key = f"captcha:{session_id}"
            raw_session = await redis_client.get(session_key)
            session_payload = safe_json_loads(raw_session)
            session_ttl = await safe_get_ttl(session_key)

        if keyword:
            keyword_lower = keyword.lower()
            if (
                keyword_lower not in ip.lower()
                and keyword_lower not in str(session_id or "").lower()
                and keyword_lower not in str(session_payload.get("text") or "").lower()
                and keyword_lower not in str(session_payload.get("pose") or "").lower()
            ):
                continue

        items.append(
            {
                "ip": ip,
                "session_id": session_id,
                "active_session_ttl_seconds": active_ttl,
                "session_ttl_seconds": session_ttl,
                "text": session_payload.get("text"),
                "pose": session_payload.get("pose"),
                "attempts": session_payload.get("attempts"),
            }
        )

    items.sort(key=lambda x: x["active_session_ttl_seconds"], reverse=True)
    return {"items": items}


@router.post("/sessions/{session_id}/expire")
async def expire_admin_handocr_session(session_id: str):
    session_key = f"captcha:{session_id}"
    raw_session = await redis_client.get(session_key)
    session_payload = safe_json_loads(raw_session)
    session_ip = session_payload.get("ip")

    deleted_count = await redis_client.delete(session_key)

    if session_ip:
        deleted_count += await redis_client.delete(f"captcha:active_session:{session_ip}")

    return {
        "success": True,
        "session_id": session_id,
        "ip": session_ip,
        "deleted_key_count": deleted_count,
    }
