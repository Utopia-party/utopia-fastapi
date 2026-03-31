from fastapi import APIRouter, Form, UploadFile, File
import httpx
import uuid
import random
import json
import redis.asyncio as redis

from core.config import settings

router = APIRouter()

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


@router.post("/handocr/start")
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


@router.post("/handocr/verify")
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

    async with httpx.AsyncClient() as client:
        files = {"image": (image.filename, image_bytes, image.content_type)}
        try:
            response = await client.post(GPU_SERVER_URL, files=files, timeout=15.0)
            response.raise_for_status()
            gpu_result = response.json()
        except Exception as e:
            return {
                "success": False,
                "message": f"AI 서버 통신 오류: {str(e)}"
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
            }
        }

    detected_pose = gpu_result.get("detected_pose")
    pose_confidence = gpu_result.get("pose_confidence")
    detected_text = gpu_result.get("detected_text")
    ocr_confidence = gpu_result.get("ocr_confidence")

    pose_ok = detected_pose == expected_pose
    text_ok = detected_text == expected_text

    if not pose_ok or not text_ok:
        session_data["attempts"] += 1
        await redis_client.setex(f"captcha:{sessionId}", 300, json.dumps(session_data))

        reasons = []
        if not pose_ok:
            reasons.append(f"손 포즈 불일치 (요구: {expected_pose} / 인식: {detected_pose})")
        if not text_ok:
            reasons.append(f"문자열 불일치 (요구: {expected_text} / 인식: {detected_text})")

        return {
            "success": False,
            "message": (
                "AI 검사에 실패했습니다.\n"
                + "\n".join(reasons)
                + (f"\n손 포즈 신뢰도: {pose_confidence:.2f}" if pose_confidence is not None else "")
                + (f"\nOCR 신뢰도: {ocr_confidence:.2f}" if ocr_confidence is not None else "")
                + "\n손 포즈와 5자리 문자+숫자가 모두 선명하게 보이도록 다시 촬영해주세요."
                + f"\n남은 기회: {MAX_ATTEMPTS - session_data['attempts']}회"
            ),
            "failureReason": {
                "type": "MISSION_MISMATCH",
                "expectedPose": expected_pose,
                "detectedPose": detected_pose,
                "expectedText": expected_text,
                "detectedText": detected_text,
                "poseConfidence": pose_confidence,
                "ocrConfidence": ocr_confidence,
                "ocrCandidates": gpu_result.get("ocr_text_candidates"),
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