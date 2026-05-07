import asyncio
import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.config import settings
from core.database import AsyncSessionLocal, get_db
from core.security import get_current_user_optional
from models.party import Party, PartyChat, PartyMember, ChatReadStatus
from models.payment import Payment
from models.user import User
from services.chat.redis_client import redis_client, REDIS_TTL
from services.chat.connection_manager import manager
from services.chat.moderation import moderate_in_background
from services.chat.read_status import _get_total_member_count, mark_all_read, mark_read_for_users
from services.chat.serializers import (
    _get_user_id_from_ws_cookie,
    _party_max_members,
    _party_member_count,
    _party_total_price,
    _serialize_member,
    _serialize_message,
    blocked_key,
    redis_msg_key,
)

router = APIRouter(prefix="/chat", tags=["chat"])


# ── REST 엔드포인트 ──────────────────────────────────────────

@router.get("/parties/{party_id}/messages")
async def get_messages(party_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PartyChat, User)
        .outerjoin(User, PartyChat.sender_id == User.id)
        .where(PartyChat.party_id == party_id, PartyChat.is_deleted == False)
        .order_by(PartyChat.created_at.desc())
        .limit(100)
    )
    rows = result.all()

    party_result = await db.execute(select(Party).where(Party.id == party_id))
    party = party_result.scalar_one_or_none()
    member_count_result = await db.execute(
        select(func.count()).select_from(PartyMember).where(
            PartyMember.party_id == party_id,
            PartyMember.status == "active",
        )
    )
    active_member_count = member_count_result.scalar() or 0
    if party:
        ldr_check = await db.execute(
            select(PartyMember).where(
                PartyMember.party_id == party_id,
                PartyMember.user_id == party.leader_id,
                PartyMember.status == "active",
            )
        )
        ldr_in = ldr_check.scalar_one_or_none() is not None
        total_members = active_member_count if ldr_in else active_member_count + 1
    else:
        total_members = active_member_count

    chat_ids = [chat.id for chat, _ in rows]
    unread_map: dict[uuid.UUID, int] = {}
    if chat_ids:
        read_counts = await db.execute(
            select(ChatReadStatus.chat_id, func.count().label("cnt"))
            .where(ChatReadStatus.chat_id.in_(chat_ids))
            .group_by(ChatReadStatus.chat_id)
        )
        read_map = {row.chat_id: row.cnt for row in read_counts}
        for chat_id in chat_ids:
            unread_map[chat_id] = max(0, total_members - read_map.get(chat_id, 0))

    return [
        _serialize_message(chat, sender, unread_map.get(chat.id, 0))
        for chat, sender in reversed(rows)
    ]


@router.get("/parties/{party_id}/info")
async def get_party_info(
    party_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_current_user_optional),
):
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
    billing_month = datetime.now(timezone.utc).strftime("%Y-%m")
    member_user_ids = [user.id for _, user in rows]
    if party.host:
        member_user_ids.append(party.host.id)
    unique_member_user_ids = list({user_id for user_id in member_user_ids})

    paid_user_ids: set[uuid.UUID] = set()
    if unique_member_user_ids:
        paid_result = await db.execute(
            select(Payment.user_id).where(
                Payment.party_id == party_id,
                Payment.billing_month == billing_month,
                Payment.status == "approved",
                Payment.user_id.in_(unique_member_user_ids),
            )
        )
        paid_user_ids = set(paid_result.scalars().all())

    members = [
        _serialize_member(
            user,
            role=member.role,
            status=member.status,
            joined_at=member.joined_at,
            payment_status="completed" if user.id in paid_user_ids else "pending",
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
                payment_status="completed" if party.host.id in paid_user_ids else "pending",
            ),
        )

    members.sort(
        key=lambda member: (
            0 if member["role"] == "leader" else 1,
            member["joined_at"] or "",
        )
    )

    service = party.service
    is_leader = False
    has_referrer_discount = False
    if current_user:
        is_leader = party.leader_id == current_user.id
        if current_user.referrer_id is not None:
            member_user_ids = {m["user_id"] for m in members}
            has_referrer_discount = str(current_user.referrer_id) in member_user_ids

    return {
        "party_id": str(party.id),
        "title": party.title,
        "status": party.status.lower() if party.status else None,
        "max_members": _party_max_members(party, service),
        "member_count": _party_member_count(party, members),
        "monthly_price": _party_total_price(party, service),
        "monthly_per_person": party.monthly_per_person,
        "quick_match_fee_rate": float(service.quick_match_fee_rate) if service and service.quick_match_fee_rate is not None else 0.0,
        "leader_discount_rate": float(service.leader_discount_rate) if service and service.leader_discount_rate is not None else None,
        "referral_discount_rate": float(service.referral_discount_rate) if service and service.referral_discount_rate is not None else None,
        "is_leader": is_leader,
        "has_referrer_discount": has_referrer_discount,
        "start_date": party.start_date.isoformat() if party.start_date else None,
        "end_date": party.end_date.isoformat() if party.end_date else None,
        "category_name": service.category if service else None,
        "service_name": service.name if service else None,
        "host_nickname": party.host.nickname if party.host else None,
        "members": members,
    }


# ── WebSocket 핸들러 ─────────────────────────────────────────

@router.websocket("/ws/{party_id}")
async def websocket_chat(
    party_id: str,
    ws: WebSocket,
    nickname: str = Query(default="익명"),
):
    jwt_user_id = _get_user_id_from_ws_cookie(ws)
    safe_user_id = jwt_user_id if jwt_user_id else "guest"

    if safe_user_id != "guest":
        try:
            party_uuid = uuid.UUID(party_id)
            user_uuid = uuid.UUID(safe_user_id)
            async with AsyncSessionLocal() as db:
                party_result = await db.execute(select(Party).where(Party.id == party_uuid))
                party = party_result.scalar_one_or_none()
                if not party:
                    await ws.close(code=4004)
                    return
                is_leader = party.leader_id == user_uuid
                if not is_leader:
                    member_result = await db.execute(
                        select(PartyMember).where(
                            PartyMember.party_id == party_uuid,
                            PartyMember.user_id == user_uuid,
                            PartyMember.status == "active",
                        )
                    )
                    if not member_result.scalar_one_or_none():
                        await ws.close(code=4003)
                        return
        except Exception as e:
            print(f"[WS AUTH ERROR] {e}")
            await ws.close(code=4000)
            return

    await manager.connect(party_id, ws, safe_user_id)

    if safe_user_id != "guest":
        await manager.send_personal(ws, {
            "type": "system_info",
            "content": "욕설·비방 시 경고 누적 후 자동 퇴장 및 신뢰도 감점이 적용됩니다. 건전한 파티 문화를 함께 만들어요.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    if safe_user_id != "guest":
        newly_read = await mark_all_read(party_id, safe_user_id)
        if newly_read:
            await manager.broadcast(party_id, {
                "type": "read_update",
                "user_id": safe_user_id,
                "chat_ids": newly_read,
            })

    try:
        while True:
            data = await ws.receive_text()

            is_blocked = await redis_client.get(blocked_key(safe_user_id))
            if is_blocked:
                await manager.send_personal(ws, {
                    "type": "error",
                    "content": "채팅이 차단되어 보낼 수 없습니다.",
                })
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

            chat_id_str = None
            try:
                async with AsyncSessionLocal() as db:
                    sender_uuid = None
                    try:
                        sender_uuid = uuid.UUID(safe_user_id)
                    except (ValueError, TypeError):
                        pass

                    new_chat = PartyChat(
                        party_id=uuid.UUID(party_id),
                        sender_id=sender_uuid,
                        message=data,
                    )
                    db.add(new_chat)
                    await db.flush()
                    chat_id_str = str(new_chat.id)
                    await db.commit()

                online_ids = manager.get_online_user_ids(party_id)
                read_targets = []
                for uid_str in online_ids:
                    if uid_str == "guest":
                        continue
                    try:
                        read_targets.append(uuid.UUID(uid_str))
                    except (ValueError, TypeError):
                        pass
                if read_targets:
                    await mark_read_for_users(uuid.UUID(chat_id_str), read_targets)

            except Exception as db_err:
                print(f"[DB ERROR] {db_err}")

            total_cnt = await _get_total_member_count(party_id)
            online_ids = manager.get_online_user_ids(party_id)
            online_count = len([u for u in online_ids if u != "guest"])
            message["chat_id"] = chat_id_str
            message["unread_count"] = max(0, total_cnt - online_count)

            await manager.broadcast(party_id, message)
            asyncio.create_task(moderate_in_background(party_id, safe_user_id, data, ws))

    except WebSocketDisconnect:
        manager.disconnect(party_id, ws, safe_user_id)
    except Exception as e:
        print(f"[WS_FATAL_ERROR] {e}")
        manager.disconnect(party_id, ws, safe_user_id)
