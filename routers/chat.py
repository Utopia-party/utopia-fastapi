import json
import asyncio
import uuid
import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import redis.asyncio as aioredis
from core.config import settings
from core.database import get_db, AsyncSessionLocal
from models.party import Party, PartyMember, PartyChat, Service
from models.user import User
from services.mypage.profile_service import _build_profile_image_url

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


def _party_max_members(party: Party, service: Service | None) -> int | None:
    if party.max_members is not None:
        return party.max_members
    if service is not None:
        return service.max_members
    return None


def _party_member_count(party: Party, members: list[dict]) -> int:
    if party.current_members is not None:
        return party.current_members
    return len(members)


def _party_total_price(party: Party, service: Service | None) -> int | None:
    if service is not None:
        return service.monthly_price

    max_members = _party_max_members(party, service)
    if party.monthly_per_person is not None and max_members:
        return party.monthly_per_person * max_members
    return None


def _safe_profile_image_url(profile_image_key: str | None) -> str | None:
    try:
        return _build_profile_image_url(profile_image_key)
    except Exception:
        return None


def _serialize_message(chat: PartyChat, sender: User | None) -> dict:
    return {
        "type": "message",
        "party_id": str(chat.party_id),
        "user_id": str(chat.sender_id) if chat.sender_id else None,
        "nickname": sender.nickname if sender else None,
        "profile_image": _safe_profile_image_url(sender.profile_image_key) if sender else None,
        "content": chat.message,
        "created_at": chat.created_at.isoformat(),
    }


def _serialize_member(
    user: User,
    *,
    role: str,
    status: str,
    joined_at: datetime | None,
) -> dict:
    return {
        "user_id": str(user.id),
        "nickname": user.nickname,
        "name": (user.name or user.nickname),
        "role": role,
        "status": status,
        "trust_score": float(user.trust_score) if user.trust_score is not None else None,
        "joined_at": joined_at.isoformat() if joined_at else None,
        "profile_image": _safe_profile_image_url(user.profile_image_key),
        "is_active": bool(user.is_active),
    }


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
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        await redis_client.rpop(redis_msg_key(party_id))
        await manager.broadcast(party_id, {
            "type": "system",
            "content": "부적절한 메시지가 삭제되었습니다.",
            "created_at": datetime.now(timezone.utc).isoformat(),
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
        parsed = [json.loads(message) for message in cached]
        if all(item.get("nickname") is not None for item in parsed if item.get("type") == "message"):
            return parsed

    result = await db.execute(
        select(PartyChat, User)
        .outerjoin(User, PartyChat.sender_id == User.id)
        .where(PartyChat.party_id == party_id, PartyChat.is_deleted == False)
        .order_by(PartyChat.created_at.desc())
        .limit(100)
    )
    rows = result.all()
    return [_serialize_message(chat, sender) for chat, sender in reversed(rows)]


@router.get("/parties/{party_id}/info")
async def get_party_info(party_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    party_result = await db.execute(
        select(Party)
        .options(selectinload(Party.host), selectinload(Party.service))
        .where(Party.id == party_id)
    )
    party = party_result.scalar_one_or_none()
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    result = await db.execute(
        select(PartyMember, User)
        .join(User, PartyMember.user_id == User.id)
        .where(PartyMember.party_id == party_id, PartyMember.status == "active")
        .order_by(PartyMember.joined_at.asc())
    )
    rows = result.all()

    members = [
        _serialize_member(
            user,
            role=member.role,
            status=member.status,
            joined_at=member.joined_at,
        )
        for member, user in rows
    ]

    if party.host and not any(member["user_id"] == str(party.host.id) for member in members):
        members.insert(
            0,
            _serialize_member(
                party.host,
                role="leader",
                status="active",
                joined_at=party.created_at,
            ),
        )

    members.sort(
        key=lambda member: (
            0 if member["role"] == "leader" else 1,
            member["joined_at"] or "",
        )
    )

    service = party.service

    return {
        "party_id": str(party.id),
        "title": party.title,
        "status": party.status.lower() if party.status else None,
        "max_members": _party_max_members(party, service),
        "member_count": _party_member_count(party, members),
        "monthly_price": _party_total_price(party, service),
        "referral_discount_rate": float(service.referral_discount_rate) if service and service.referral_discount_rate is not None else None,
        "monthly_per_person": party.monthly_per_person,
        "start_date": party.start_date.isoformat() if party.start_date else None,
        "end_date": party.end_date.isoformat() if party.end_date else None,
        "category_name": service.category if service else None,
        "service_name": service.name if service else None,
        "host_nickname": party.host.nickname if party.host else None,
        "members": members,
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
        while True:
            data = await ws.receive_text()

            is_blocked = await redis_client.get(blocked_key(party_id, safe_user_id))
            if is_blocked:
                await manager.send_personal(ws, {"type": "error", "content": "채팅이 차단되어 보낼 수 없습니다."})
                continue

            now = datetime.now(timezone.utc).isoformat()  
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
    except Exception as e:
        print(f"[WS_FATAL_ERROR] {e}")
        manager.disconnect(party_id, ws)
