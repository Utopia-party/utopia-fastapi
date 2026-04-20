import asyncio
import secrets
import uuid

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from core.database import AsyncSessionLocal
from core.redis_client import redis_client
from core.security import require_user
from models.user import User
from services.notification_service import get_unread_notification_count_service
from services.notification_ws_service import notification_connection_manager


router = APIRouter(tags=["notifications-ws"])

WS_TOKEN_TTL = 30  # 30초 안에 WS 연결 안 하면 만료


@router.post("/api/ws-token")
async def issue_ws_token(current_user: User = Depends(require_user)):
    """WebSocket 연결용 단기 토큰 발급 (30초 유효)"""
    token = secrets.token_urlsafe(32)
    await redis_client.setex(f"ws_token:{token}", WS_TOKEN_TTL, str(current_user.id))
    return {"token": token}


async def get_user_id_from_ws_token(token: str) -> uuid.UUID | None:
    user_id_str = await redis_client.get(f"ws_token:{token}")
    if not user_id_str:
        return None
    # 1회용: 사용 후 즉시 삭제
    await redis_client.delete(f"ws_token:{token}")
    try:
        return uuid.UUID(user_id_str)
    except ValueError:
        return None


@router.websocket("/ws/notifications")
async def notifications_websocket(websocket: WebSocket):
    token = websocket.query_params.get("token")

    if not token:
        await websocket.accept()
        await websocket.close(code=4401)
        return

    user_id = await get_user_id_from_ws_token(token)

    if user_id is None:
        await websocket.accept()
        await websocket.close(code=4401)
        return

    await notification_connection_manager.connect(user_id, websocket)

    try:
        async with AsyncSessionLocal() as db:
            unread_count = await get_unread_notification_count_service(
                db=db,
                user_id=user_id,
            )

        await websocket.send_json(
            {
                "type": "connected",
                "message": "알림 웹소켓 연결 성공",
                "unread_count": unread_count,
            }
        )

        while True:
            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30,
                )
                await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        notification_connection_manager.disconnect(user_id, websocket)
    except Exception:
        notification_connection_manager.disconnect(user_id, websocket)
        try:
            await websocket.close()
        except Exception:
            pass
