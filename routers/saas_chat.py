from typing import Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from core.saas_key_auth import verify_saas_key
from schemas.sass import SaasKeyContext
from services.chat.moderation import check_message

router = APIRouter(prefix="/saas/chat", tags=["SaaS Chat Filter"])


async def _require_chat_key(request: Request) -> Optional[SaasKeyContext]:
    return await verify_saas_key(request, required_service="chat_filter")


class ChatCheckRequest(BaseModel):
    content: str


@router.post("/check")
async def chat_check(
    payload: ChatCheckRequest,
    saas_ctx: Optional[SaasKeyContext] = Depends(_require_chat_key),
):
    result = await check_message(payload.content)
    return result
