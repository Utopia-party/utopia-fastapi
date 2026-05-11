from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from core.saas_key_auth import verify_saas_key
from schemas.sass import SaasKeyContext
from routers.captcha import _start_captcha_logic, _verify_captcha_logic

router = APIRouter(tags=["SaaS Captcha L2"])


async def _require_l2_key(request: Request) -> Optional[SaasKeyContext]:
    return await verify_saas_key(request, required_service="captcha_l2")


@router.post("/saas/captcha/handocr/start")
async def saas_start_captcha(
    request: Request,
    saas_ctx: Optional[SaasKeyContext] = Depends(_require_l2_key),
):
    return await _start_captcha_logic(request)


@router.post("/saas/captcha/handocr/verify")
async def saas_verify_captcha(
    request: Request,
    sessionId: str = Form(...),
    image: UploadFile = File(...),
    saas_ctx: Optional[SaasKeyContext] = Depends(_require_l2_key),
):
    return await _verify_captcha_logic(request, sessionId, image)
