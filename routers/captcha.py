from fastapi import APIRouter, File, Form, Request, UploadFile
import httpx
import uuid
import random
import json
import redis.asyncio as redis
import traceback
import logging
from typing import Any

from core.config import settings
from schemas.captcha import (
    CaptchaChallengeResponse,
    CaptchaInitRequest,
    CaptchaInitResponse,
    CaptchaStatusResponse,
    CaptchaVerifyRequest,
    CaptchaVerifyResponse,
)
from services.captcha_service import (
    get_captcha_status,
    get_challenge,
    initiate_captcha,
    verify_challenge,
)

router = APIRouter()
logger = logging.getLogger("handocr-api")


# ─────────────────────────────────────────────
# 1차 캡챠 API
# ─────────────────────────────────────────────


@router.post("/init", response_model=CaptchaInitResponse)
async def captcha_init(payload: CaptchaInitRequest, request: Request):
    return await initiate_captcha(payload, request)


@router.get("/challenge", response_model=CaptchaChallengeResponse)
async def captcha_challenge(session_id: str, request: Request):
    return await get_challenge(session_id, request)


@router.post("/verify", response_model=CaptchaVerifyResponse)
async def captcha_verify(payload: CaptchaVerifyRequest, request: Request):
    return await verify_challenge(payload, request)


@router.get("/status", response_model=CaptchaStatusResponse)
async def captcha_status(request: Request):
    return await get_captcha_status(request)


redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
GPU_SERVER_URL = settings.GPU_SERVER_URL

ALL_POSES = [
    "주먹 ✊",
    "손바닥 🖐️",
    "브이 ✌️",
    "따봉 👍",
]

MAX_ATTEMPTS = 5


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


@router.post("/captcha/handocr/start")
async def start_captcha():
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    random_text = "".join(random.choice(chars) for _ in range(5))
    random_pose = random.choice(ALL_POSES)

    session_id = str(uuid.uuid4())
    session_data = {
        "text": random_text,
        "pose": random_pose,
        "attempts": 0
    }

    await redis_client.setex(f"captcha:{session_id}", 300, json.dumps(session_data))

    return {
        "sessionId": session_id,
        "text": random_text,
        "pose": random_pose
    }


@router.post("/captcha/handocr/verify")
async def verify_captcha(sessionId: str = Form(...), image: UploadFile = File(...)):
    session_str = await redis_client.get(f"captcha:{sessionId}")
    if not session_str:
        return {
            "success": False,
            "message": "유효하지 않거나 5분이 지나 만료된 세션입니다. 새로고침 후 다시 시작해주세요."
        }

    session_data = json.loads(session_str)
    expected_pose = session_data["pose"]
    expected_text = session_data["text"]

    if session_data.get("attempts", 0) >= MAX_ATTEMPTS:
        await redis_client.delete(f"captcha:{sessionId}")
        return {
            "success": False,
            "message": "실패 횟수(5회)를 초과했습니다. 새로고침하여 처음부터 다시 시도해주세요."
        }

    image_bytes = await image.read()
    if not image_bytes:
        session_data["attempts"] += 1
        await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
        return {
            "success": False,
            "message": (
                "업로드된 이미지가 비어 있습니다.\n"
                "사진을 다시 찍거나 다른 파일을 업로드해주세요.\n"
                f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
            ),
            "failureReason": {
                "type": "EMPTY_IMAGE",
                "expectedPose": expected_pose,
                "expectedText": expected_text,
            }
        }

    gpu_result = None
    gpu_status_code = None
    gpu_raw_text = None

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
            except Exception as json_error:
                logger.exception("GPU json parse failed")
                session_data["attempts"] += 1
                await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
                return {
                    "success": False,
                    "message": (
                        "AI 서버 응답을 해석하지 못했습니다.\n"
                        "AI 서버가 JSON이 아닌 응답을 반환했습니다.\n"
                        f"상세: {repr(json_error)}\n"
                        f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
                    ),
                    "failureReason": {
                        "type": "GPU_JSON_PARSE_FAILED",
                        "expectedPose": expected_pose,
                        "expectedText": expected_text,
                        "statusCode": gpu_status_code,
                        "rawText": gpu_raw_text,
                        "detail": repr(json_error),
                        "gpuServerUrl": GPU_SERVER_URL,
                    }
                }

            logger.info("GPU json=%s", gpu_result)

        except httpx.ConnectTimeout as e:
            logger.exception("GPU connect timeout")
            session_data["attempts"] += 1
            await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
            return {
                "success": False,
                "message": (
                    "AI 서버 연결 시간이 초과되었습니다.\n"
                    "GPU 서버가 응답 가능한 상태인지 확인해주세요.\n"
                    f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
                ),
                "failureReason": {
                    "type": "GPU_CONNECT_TIMEOUT",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            }

        except httpx.ConnectError as e:
            logger.exception("GPU connect error")
            session_data["attempts"] += 1
            await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
            return {
                "success": False,
                "message": (
                    "AI 서버 연결에 실패했습니다.\n"
                    "GPU 서버가 꺼져 있거나, 주소/포트가 잘못되었거나, 네트워크에 문제가 있을 수 있습니다.\n"
                    f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
                ),
                "failureReason": {
                    "type": "GPU_CONNECT_ERROR",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            }

        except httpx.ReadTimeout as e:
            logger.exception("GPU read timeout")
            session_data["attempts"] += 1
            await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
            return {
                "success": False,
                "message": (
                    "AI 서버가 사진을 분석하는 데 시간이 너무 오래 걸리고 있습니다.\n"
                    "서버는 살아 있지만 응답이 지연되고 있을 수 있습니다.\n"
                    f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
                ),
                "failureReason": {
                    "type": "GPU_READ_TIMEOUT",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            }

        except httpx.HTTPStatusError as e:
            logger.exception("GPU HTTP status error")
            session_data["attempts"] += 1
            await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))

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

            return {
                "success": False,
                "message": (
                    f"{message}\n"
                    f"HTTP 상태 코드: {status_code}\n"
                    f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
                ),
                "failureReason": {
                    "type": "GPU_HTTP_ERROR",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "statusCode": status_code,
                    "responseText": response_text,
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            }

        except httpx.RequestError as e:
            logger.exception("GPU request error")
            session_data["attempts"] += 1
            await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
            return {
                "success": False,
                "message": (
                    "AI 서버 요청 처리 중 네트워크 오류가 발생했습니다.\n"
                    f"상세: {repr(e)}\n"
                    f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
                ),
                "failureReason": {
                    "type": "GPU_REQUEST_ERROR",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            }

        except Exception as e:
            logger.exception("GPU request/parse failed")
            traceback.print_exc()
            session_data["attempts"] += 1
            await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
            return {
                "success": False,
                "message": (
                    "AI 서버 통신 처리 중 알 수 없는 오류가 발생했습니다.\n"
                    f"상세: {repr(e)}\n"
                    f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
                ),
                "failureReason": {
                    "type": "GPU_UNKNOWN_ERROR",
                    "expectedPose": expected_pose,
                    "expectedText": expected_text,
                    "detail": repr(e),
                    "gpuServerUrl": GPU_SERVER_URL,
                }
            }

    if not isinstance(gpu_result, dict):
        session_data["attempts"] += 1
        await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
        return {
            "success": False,
            "message": (
                "AI 서버 응답 형식이 올바르지 않습니다.\n"
                f"남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
            ),
            "failureReason": {
                "type": "GPU_INVALID_RESPONSE",
                "expectedPose": expected_pose,
                "expectedText": expected_text,
                "gpuResultType": str(type(gpu_result)),
                "gpuServerUrl": GPU_SERVER_URL,
            }
        }

    if not gpu_result.get("success"):
        session_data["attempts"] += 1
        await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))
        remaining_attempts = MAX_ATTEMPTS - session_data["attempts"]

        return {
            "success": False,
            "message": build_ai_failure_message(gpu_result, remaining_attempts),
            "failureReason": {
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
        }

    detected_pose = gpu_result.get("detected_pose")
    pose_confidence = safe_float(gpu_result.get("pose_confidence"))
    detected_text = gpu_result.get("detected_text")
    ocr_confidence = safe_float(gpu_result.get("ocr_confidence"))
    ocr_low_confidence = gpu_result.get("ocr_low_confidence", False)

    pose_ok = detected_pose == expected_pose
    text_ok = detected_text == expected_text

    if not pose_ok or not text_ok:
        session_data["attempts"] += 1
        await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))

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
                reasons.append(f"문자열 불일치 (요구: {expected_text} / 인식: {detected_text})")

        message = "AI 검사에 실패했습니다.\n" + "\n".join(reasons)

        if pose_confidence is not None:
            message += f"\n손 포즈 신뢰도: {pose_confidence:.2f}"

        if ocr_confidence is not None:
            message += f"\nOCR 신뢰도: {ocr_confidence:.2f}"

        message += "\n손 포즈와 5자리 문자+숫자가 모두 선명하게 보이도록 다시 촬영해주세요."
        message += f"\n남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"

        return {
            "success": False,
            "message": message,
            "failureReason": {
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
        }

    pass_token = str(uuid.uuid4())
    await redis_client.setex(f"captcha_pass:{pass_token}", 180, "PASSED")
    await redis_client.delete(f"captcha:{sessionId}")

    return {
        "success": True,
        "message": "인증이 완료되었습니다.",
        "passToken": pass_token
    }
