import json
import asyncio
import uuid
import httpx
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import redis.asyncio as aioredis
from core.config import settings
from core.database import get_db, AsyncSessionLocal
from models.party import Party, PartyMember, PartyChat
from models.user import User
 
router = APIRouter(prefix="/chat", tags=["chat"])
 
# Redis 클라이언트 설정
redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
 
REDIS_TTL = 60 * 60 * 24 * 3
OLLAMA_URL = settings.OLLAMA_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
 
 
# Redis 키 생성 함수들
def warn_key(party_id: str, user_id: str) -> str:
    return f"warn:{party_id}:{user_id}"
 
def redis_msg_key(party_id: str) -> str:
    return f"chat:party:{party_id}:messages"
 
def blocked_key(party_id: str, user_id: str) -> str:
    return f"blocked:{party_id}:{user_id}"
 
 
# 메시지 필터링 (Ollama 연동)
async def check_message(content: str) -> dict:
    prompt = f"""채팅 메시지에 욕설, 비속어, 혐오 표현이 있는지 판단하세요.
메시지: "{content}"
JSON으로만 응답하세요:
{{"violation": true/false, "severe": true/false, "reason": "이유"}}"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
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
 
 
# 웹소켓 연결 관리 클래스
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
        if party_id in self.active:
            for ws in list(self.active[party_id]):
                try:
                    await ws.send_text(msg_str)
                except Exception:
                    self.disconnect(party_id, ws)
 
    async def send_personal(self, ws: WebSocket, message: dict):
        try:
            await ws.send_text(json.dumps(message, ensure_ascii=False))
        except Exception:
            pass
 
 
manager = ConnectionManager()
 
 
# 백그라운드 모더레이션 로직
async def moderate_in_background(party_id: str, user_id: str, content: str, ws: WebSocket):
    moderation = await check_message(content)
    if moderation["severe"]:
        await redis_client.set(blocked_key(party_id, user_id), "1", ex=REDIS_TTL)
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
            await manager.send_personal(ws, {"type": "error", "content": "🚫 경고 3회 누적으로 채팅이 차단되었습니다."})
        else:
            await manager.send_personal(ws, {"type": "warning", "content": f"⚠️ 경고 {warn_count}/3회: 부적절한 표현 감지."})
 
 
# ── 일반 API 엔드포인트 ──────────────────────────────────────
 
@router.get("/parties/{party_id}/messages")
async def get_messages(party_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    cached = await redis_client.lrange(redis_msg_key(str(party_id)), 0, -1)
    if cached:
        return [json.loads(m) for m in cached]
    result = await db.execute(
        select(PartyChat)
        .where(PartyChat.party_id == party_id, PartyChat.is_deleted == False)
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
    # Party + host + service 한 번에 로드
    result = await db.execute(
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.service),
            selectinload(Party.members).selectinload(PartyMember.user),
        )
        .where(Party.id == party_id)
    )
    party = result.scalar_one_or_none()
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")
 
    # active 멤버만 필터링 + profile_image 포함
    active_members = [
        {
            "user_id": str(m.user.id),
            "nickname": m.user.nickname,
            "role": m.role,
            "status": m.status,
            "profile_image": getattr(m.user, "profile_image", None),
        }
        for m in party.members
        if m.status == "active" and m.user is not None
    ]
 
    # 1인당 비용: Party.monthly_per_person 우선, 없으면 Service.monthly_price / max_members 계산
    per_person = party.monthly_per_person
    total_price = None
    if party.service:
        total_price = party.service.monthly_price
        if per_person is None and party.max_members:
            per_person = round(total_price / party.max_members)
 
    return {
        "party_id": str(party.id),
        "title": party.title,
        "description": party.description,
        "status": party.status,
        "max_members": party.max_members,
        "member_count": party.current_members or len(active_members),
        "monthly_price": total_price,
        "monthly_per_person": per_person,
        "start_date": party.start_date.isoformat() if party.start_date else None,
        "end_date": party.end_date.isoformat() if party.end_date else None,
        "category_name": party.service.category if party.service else None,
        "service_name": party.service.name if party.service else None,
        "host_nickname": party.host.nickname if party.host else None,
        "members": active_members,
    }
 
 
# ── 웹소켓 메인 핸들러 ────────────────────────────────────────
 
@router.websocket("/ws/{party_id}")
async def websocket_chat(
    party_id: str,
    ws: WebSocket,
    nickname: str = Query(default="익명"),
    user_id: str = Query(default="guest"),
):
    safe_user_id = user_id
    if user_id == "undefined" or not user_id:
        safe_user_id = "guest"
 
    await manager.connect(party_id, ws)
 
    try:
        await manager.broadcast(party_id, {
            "type": "system",
            "content": f"{nickname}님이 입장했습니다.",
            "created_at": datetime.now().isoformat(),
        })
 
        while True:
            data = await ws.receive_text()
 
            is_blocked = await redis_client.get(blocked_key(party_id, safe_user_id))
            if is_blocked:
                await manager.send_personal(ws, {"type": "error", "content": "채팅이 차단되어 보낼 수 없습니다."})
                continue
 
            now = datetime.now().isoformat()
            message = {
                "type": "message",
                "party_id": party_id,
                "user_id": safe_user_id,
                "nickname": nickname,
                "content": data,
                "created_at": now,
            }
 
            key = redis_msg_key(party_id)
            await redis_client.rpush(key, json.dumps(message, ensure_ascii=False))
            await redis_client.ltrim(key, -200, -1)
            await redis_client.expire(key, REDIS_TTL)
 
            try:
                async with AsyncSessionLocal() as db:
                    sender_uuid = None
                    try:
                        sender_uuid = uuid.UUID(safe_user_id)
                    except (ValueError, TypeError):
                        sender_uuid = None
 
                    new_chat = PartyChat(
                        party_id=uuid.UUID(party_id),
                        sender_id=sender_uuid,
                        message=data,
                    )
                    db.add(new_chat)
                    await db.commit()
            except Exception as db_err:
                print(f"[DB ERROR] {db_err}")
 
            await manager.broadcast(party_id, message)
            asyncio.create_task(moderate_in_background(party_id, safe_user_id, data, ws))
 
    except WebSocketDisconnect:
        manager.disconnect(party_id, ws)
        await manager.broadcast(party_id, {
            "type": "system",
            "content": f"{nickname}님이 퇴장했습니다.",
            "created_at": datetime.now().isoformat(),
        })
    except Exception as e:
        print(f"[WS_FATAL_ERROR] {e}")
        manager.disconnect(party_id, ws)
