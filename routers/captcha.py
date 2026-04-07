import json
import logging
import random
import traceback
import uuid
from typing import Any, List, Optional

import httpx
import redis.asyncio as redis
from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException

from core.config import settings
from schemas.captcha import (
    CaptchaChallengeResponse,
    CaptchaInitRequest,
    CaptchaInitResponse,
    CaptchaStatusResponse,
    CaptchaVerifyRequest,
    CaptchaVerifyResponse,
)
from services import captcha_service

router = APIRouter()
logger = logging.getLogger("captcha-api")

# Redis 및 설정 (Hand OCR 전용)
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
GPU_SERVER_URL = settings.GPU_SERVER_URL
ALL_POSES = ["주먹 ✊", "손바닥 🖐️", "브이 ✌️", "따봉 👍"]
MAX_ATTEMPTS = 5

# ─────────────────────────────────────────────
# 1. 이모지 그리드 API (captcha_service 연동)
# ─────────────────────────────────────────────

@router.get("/status", response_model=CaptchaStatusResponse)
async def captcha_status(request: Request):
    return await captcha_service.get_captcha_status(request)

@router.post("/init", response_model=CaptchaInitResponse)
async def captcha_init(payload: CaptchaInitRequest, request: Request):
    return await captcha_service.initiate_captcha(payload, request)

@router.get("/challenge", response_model=CaptchaChallengeResponse)
async def captcha_challenge(session_id: str, request: Request):
    return await captcha_service.get_challenge(session_id, request)

@router.post("/verify", response_model=CaptchaVerifyResponse)
async def captcha_verify(payload: CaptchaVerifyRequest, request: Request):
    return await captcha_service.verify_challenge(payload, request)

# ─────────────────────────────────────────────
# 2. Hand OCR API
# ─────────────────────────────────────────────

def build_ai_failure_message(gpu_result: dict, remaining_attempts: int) -> str:
    error_code = gpu_result.get("error_code", "UNKNOWN_ERROR")
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
    }
    msg = title_map.get(error_code, f"AI 검사에 실패했습니다. {gpu_result.get('message', '')}")
    return f"{msg}\n남은 기회: {remaining_attempts}회"

def safe_float(value: Any):
    try: return float(value) if value is not None else None
    except: return None

@router.post("/handocr/start")
async def start_handocr():
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    random_text = "".join(random.choice(chars) for _ in range(5))
    random_pose = random.choice(ALL_POSES)
    session_id = str(uuid.uuid4())
    session_data = {"text": random_text, "pose": random_pose, "attempts": 0}
    await redis_client.setex(f"captcha:{session_id}", 300, json.dumps(session_data))
    return {"sessionId": session_id, "text": random_text, "pose": random_pose}

@router.post("/handocr/verify")
async def verify_handocr(sessionId: str = Form(...), image: UploadFile = File(...)):
    session_str = await redis_client.get(f"captcha:{sessionId}")
    if not session_str:
        return {"success": False, "message": "유효하지 않거나 만료된 세션입니다."}

    session_data = json.loads(session_str)
    if session_data.get("attempts", 0) >= MAX_ATTEMPTS:
        await redis_client.delete(f"captcha:{sessionId}")
        return {"success": False, "message": "실패 횟수 초과"}

    image_bytes = await image.read()
    timeout = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            files = {"image": (image.filename or "upload.jpg", image_bytes, image.content_type)}
            response = await client.post(GPU_SERVER_URL, files=files)
            response.raise_for_status()
            gpu_result = response.json()
        except Exception as e:
            session_data["attempts"] += 1
            await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
            return {"success": False, "message": f"AI 서버 통신 오류: {str(e)}"}

    if not gpu_result.get("success"):
        session_data["attempts"] += 1
        await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
        return {"success": False, "message": build_ai_failure_message(gpu_result, MAX_ATTEMPTS - session_data["attempts"])}

    # 검증 로직
    detected_pose = gpu_result.get("detected_pose")
    detected_text = gpu_result.get("detected_text")
    
    if detected_pose == session_data["pose"] and detected_text == session_data["text"]:
        pass_token = str(uuid.uuid4())
        await redis_client.setex(f"captcha_pass:{pass_token}", 180, "PASSED")
        await redis_client.delete(f"captcha:{sessionId}")
        return {"success": True, "message": "인증 성공", "passToken": pass_token}
    
    session_data["attempts"] += 1
    await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
    return {"success": False, "message": "정보가 일치하지 않습니다.", "remaining_attempts": MAX_ATTEMPTS - session_data["attempts"]}