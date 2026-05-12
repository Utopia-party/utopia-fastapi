from datetime import datetime

from fastapi import WebSocket
from jose import JWTError, jwt

from core.config import settings
from models.party import PartyChat, Service, Party
from models.user import User
from services.mypage.profile_service import _build_profile_image_url
import uuid


def warn_key(party_id: str, user_id: str) -> str:
    return f"warn:{party_id}:{user_id}"


def redis_msg_key(party_id: str) -> str:
    return f"chat:party:{party_id}:messages"


def blocked_key(user_id: str) -> str:
    return f"blocked:user:{user_id}"


def _safe_profile_image_url(profile_image_key: str | None) -> str | None:
    try:
        return _build_profile_image_url(profile_image_key)
    except Exception:
        return None


def _serialize_message(chat: PartyChat, sender: User | None, unread_count: int = 0) -> dict:
    msg_type = chat.message_type if chat.message_type else "text"
    if msg_type == "system":
        return {
            "type": "system",
            "chat_id": str(chat.id),
            "party_id": str(chat.party_id),
            "content": chat.message,
            "created_at": chat.created_at.isoformat(),
        }
    return {
        "type": "message",
        "chat_id": str(chat.id),
        "party_id": str(chat.party_id),
        "user_id": str(chat.sender_id) if chat.sender_id else None,
        "nickname": sender.nickname if sender else None,
        "profile_image": _safe_profile_image_url(sender.profile_image_key) if sender else None,
        "content": chat.message,
        "created_at": chat.created_at.isoformat(),
        "unread_count": unread_count,
    }


def _serialize_member(
    user: User,
    *,
    role: str,
    status: str,
    joined_at: datetime | None,
    payment_status: str | None = None,
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
        "payment_status": payment_status,
        "is_active": bool(user.is_active),
    }


def _party_max_members(party: Party, service: Service | None) -> int | None:
    if party.max_members is not None:
        return party.max_members
    if service is not None:
        return service.max_members
    return None


def _party_member_count(party: Party, members: list[dict]) -> int:
    return len(members)


def _party_total_price(party: Party, service: Service | None) -> int | None:
    if service is not None:
        return service.original_price or service.monthly_price
    max_members = _party_max_members(party, service)
    if party.monthly_per_person is not None and max_members:
        return party.monthly_per_person * max_members
    return None


async def save_system_message(party_id: str, content: str) -> str | None:
    """시스템 메시지를 DB에 저장하고 chat_id를 반환합니다."""
    import uuid as _uuid
    from core.database import AsyncSessionLocal
    from models.party import PartyChat
    try:
        party_uuid = _uuid.UUID(party_id)
        async with AsyncSessionLocal() as db:
            new_chat = PartyChat(
                party_id=party_uuid,
                sender_id=None,
                message=content,
                message_type="system",
            )
            db.add(new_chat)
            await db.flush()
            chat_id = str(new_chat.id)
            await db.commit()
            return chat_id
    except Exception as e:
        print(f"[SYSTEM MSG SAVE ERROR] {e}")
        return None


def _get_user_id_from_ws_cookie(ws: WebSocket) -> str | None:
    access_token = ws.cookies.get("access_token")
    if not access_token:
        return None
    try:
        payload = jwt.decode(
            access_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        if payload.get("type") != "access":
            return None
        user_id_str = payload.get("sub", "")
        uuid.UUID(user_id_str)
        return user_id_str
    except (JWTError, ValueError):
        return None
