import json
import asyncio
import uuid
import httpx
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import redis.asyncio as aioredis
from core.config import settings
from core.database import get_db
from models.party import Party, PartyMember, PartyChat
from models.user import User

router = APIRouter(prefix="/chat", tags=["chat"])

redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

REDIS_TTL = 60 * 60 * 24 * 3
OLLAMA_URL = settings.OLLAMA_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL


def warn_key(party_id: str, user_id: str) -> str:
    return f"warn:{party_id}:{user_id}"

def redis_msg_key(party_id: str) -> str:
    return f"chat:party:{party_id}:messages"

def blocked_key(party_id: str, user_id: str) -> str:
    return f"blocked:{party_id}:{user_id}"


async def check_message(content: str) -> dict:
    prompt = f"""채팅 메시지에 욕설, 비속어, 혐오 표현이 있는지 판단하세요.

메시지: "{content}"

JSON으로만 응답하세요:
{{"violation": true/false, "severe": true/false, "reason": "이유"}}"""

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            text = resp.json()["message"]["content"].strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            parsed = json.loads(text)
            return {
                "violation": parsed.get("violation", False),
                "severe": parsed.get("severe", False),
                "reason": parsed.get("reason", ""),
            }
    except Exception:
        return {"violation": False, "severe": False, "reason": ""}


class ConnectionManager:
    def __init__(self):
        self.active: dict[str, list[WebSocket]] = {}

    async def connect(self, party_id: str, ws: WebSocket):
        await ws.accept()
        self.active.setdefault(party_id, []).append(ws)

    def disconnect(self, party_id: str, ws: WebSocket):
        if party_id in self.active:
            try:
                self.active[party_id].remove(ws)
            except ValueError:
                pass

    async def broadcast(self, party_id: str, message: dict):
        msg_str = json.dumps(message, ensure_ascii=False)
        for ws in list(self.active.get(party_id, [])):
            try:
                await ws.send_text(msg_str)
            except Exception:
                pass

    async def send_personal(self, ws: WebSocket, message: dict):
        try:
            await ws.send_text(json.dumps(message, ensure_ascii=False))
        except Exception:
            pass


manager = ConnectionManager()


async def moderate_in_background(
    party_id: str, user_id: str, content: str, ws: WebSocket, db: AsyncSession
):
    moderation = await check_message(content)

    if moderation["severe"]:
        await redis_client.set(blocked_key(party_id, user_id), "1", ex=REDIS_TTL)
        await db.execute(
            PartyChat.__table__.update()
            .where(PartyChat.party_id == uuid.UUID(party_id))
            .where(PartyChat.sender_id == uuid.UUID(user_id))
            .where(PartyChat.is_flagged == False)  # noqa
            .values(is_flagged=True, flag_reason=moderation["reason"], moderation_status="blocked")
        )
        await db.commit()
        await manager.send_personal(ws, {
            "type": "error",
            "content": f"🚫 심각한 욕설이 감지되어 차단되었습니다. ({moderation['reason']})",
            "created_at": datetime.now().isoformat(),
        })
        await redis_client.rpop(redis_msg_key(party_id))
        await manager.broadcast(party_id, {
            "type": "system",
            "content": "부적절한 메시지가 삭제되었습니다.",
            "created_at": datetime.now().isoformat(),
        })

    elif moderation["violation"]:
        key = warn_key(party_id, user_id)
        warn_count = await redis_client.incr(key)
        await redis_client.expire(key, REDIS_TTL)

        if warn_count >= 3:
            await redis_client.set(blocked_key(party_id, user_id), "1", ex=REDIS_TTL)
            await manager.send_personal(ws, {
                "type": "error",
                "content": "🚫 경고 3회 누적으로 채팅이 차단되었습니다.",
                "created_at": datetime.now().isoformat(),
            })
            await redis_client.rpop(redis_msg_key(party_id))
            await manager.broadcast(party_id, {
                "type": "system",
                "content": "부적절한 메시지가 삭제되었습니다.",
                "created_at": datetime.now().isoformat(),
            })
        else:
            await manager.send_personal(ws, {
                "type": "warning",
                "content": f"⚠️ 경고 {warn_count}/3회: 부적절한 표현이 감지되었습니다.",
                "created_at": datetime.now().isoformat(),
            })


@router.get("/parties/{party_id}/messages")
async def get_messages(party_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    cached = await redis_client.lrange(redis_msg_key(str(party_id)), 0, -1)
    if cached:
        return [json.loads(m) for m in cached]

    result = await db.execute(
        select(PartyChat)
        .where(PartyChat.party_id == party_id, PartyChat.is_deleted == False)  # noqa
        .order_by(PartyChat.created_at.desc())
        .limit(100)
    )
    chats = result.scalars().all()
    return [
        {
            "type": "message",
            "party_id": str(c.party_id),
            "user_id": str(c.sender_id),
            "content": c.message,
            "created_at": c.created_at.isoformat(),
        }
        for c in reversed(chats)
    ]


@router.get("/parties/{party_id}/info")
async def get_party_info(party_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PartyMember, User)
        .join(User, PartyMember.user_id == User.id)
        .where(PartyMember.party_id == party_id, PartyMember.status == "active")
    )
    rows = result.all()
    party = await db.get(Party, party_id)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    members = []
    for member, user in rows:
        members.append({
            "user_id": str(user.id),
            "nickname": user.nickname,
            "role": member.role,
            "status": member.status,
        })
    return {"party_id": str(party_id), "title": party.title, "members": members}


@router.websocket("/ws/{party_id}")
async def websocket_chat(
    party_id: str,
    ws: WebSocket,
    nickname: str = "익명",
    user_id: str = "guest",
    db: AsyncSession = Depends(get_db),
):
    await manager.connect(party_id, ws)
    await manager.broadcast(party_id, {
        "type": "system",
        "content": f"{nickname}님이 입장했습니다.",
        "created_at": datetime.now().isoformat(),
    })

    try:
        while True:
            data = await ws.receive_text()

            is_blocked = await redis_client.get(blocked_key(party_id, user_id))
            if is_blocked:
                await manager.send_personal(ws, {
                    "type": "error",
                    "content": "채팅이 차단되어 메시지를 보낼 수 없습니다.",
                    "created_at": datetime.now().isoformat(),
                })
                continue

            now = datetime.now().isoformat()
            message = {
                "type": "message",
                "party_id": party_id,
                "user_id": user_id,
                "nickname": nickname,
                "content": data,
                "created_at": now,
            }

            key = redis_msg_key(party_id)
            await redis_client.rpush(key, json.dumps(message, ensure_ascii=False))
            await redis_client.ltrim(key, -200, -1)
            await redis_client.expire(key, REDIS_TTL)

            try:
                chat = PartyChat(
                    party_id=uuid.UUID(party_id),
                    sender_id=uuid.UUID(user_id),
                    message=data,
                )
                db.add(chat)
                await db.commit()
            except Exception:
                await db.rollback()

            await manager.broadcast(party_id, message)

            asyncio.create_task(
                moderate_in_background(party_id, user_id, data, ws, db)
            )

    except WebSocketDisconnect:
        manager.disconnect(party_id, ws)
        await manager.broadcast(party_id, {
            "type": "system",
            "content": f"{nickname}님이 퇴장했습니다.",
            "created_at": datetime.now().isoformat(),
        })
