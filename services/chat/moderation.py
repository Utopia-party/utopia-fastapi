import json
import uuid
import httpx
from datetime import datetime, timezone, timedelta

from fastapi import WebSocket
from sqlalchemy import select, update

from core.config import settings
from core.database import AsyncSessionLocal
from models.party import Party, PartyMember, PartyChat
from models.user import User
from models.refresh_token import RefreshToken
from services.chat.connection_manager import manager
from services.chat.serializers import warn_key, redis_msg_key, blocked_key

from services.chat.redis_client import redis_client, REDIS_TTL


OLLAMA_URL = settings.OLLAMA_URL
OLLAMA_MODEL = settings.OLLAMA_MODEL
ML_SERVER_URL = settings.ML_SERVER_URL

LABEL_KO = {
    "hate": "혐오/심한 욕설",
    "offensive": "부적절한 표현",
}


# ── 3단계 탐지 파이프라인 ────────────────────────────────────

async def check_message(content: str) -> dict:
    from routers.admin_moderation_config import get_config
    config = await get_config()
    stripped = content.strip()

    if config.get("stage1_enabled", True):
        whitelist = config.get("whitelist", [])
        blacklist = config.get("blacklist", [])
        has_blacklist = any(w in stripped for w in blacklist)
        has_whitelist = any(w in stripped for w in whitelist)
        if has_blacklist:
            return {"violation": True, "severe": True, "reason": "욕설 축약어", "stage": 1, "score": None}
        if has_whitelist:
            return {"violation": False, "severe": False, "reason": "", "stage": 1, "score": None}

    if config.get("stage2_enabled", True) and ML_SERVER_URL:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(ML_SERVER_URL, json={"content": stripped})
                ml = resp.json()
                label = ml["label"]
                score = ml["score"]
                pass_t = config.get("stage2_pass_threshold", 0.75)
                block_t = config.get("stage2_block_threshold", 0.97)
                if label == "none" and score >= pass_t:
                    return {"violation": False, "severe": False, "reason": "", "stage": 2, "score": score}
                if label != "none" and score >= block_t:
                    return {
                        "violation": True,
                        "severe": label == "hate",
                        "reason": LABEL_KO.get(label, label),
                        "stage": 2,
                        "score": score,
                    }
        except Exception:
            pass

    if config.get("stage3_enabled", True):
        return await _check_message_ollama(content, config)

    return {"violation": False, "severe": False, "reason": "", "stage": 0, "score": None}


async def _check_message_ollama(content: str, config: dict) -> dict:
    examples = config.get("ollama_prompt_examples", [])
    none_ex = [e["text"] for e in examples if e["label"] == "none"]
    offensive_ex = [e["text"] for e in examples if e["label"] == "offensive"]

    none_str = ", ".join(f'"{t}"' for t in none_ex) if none_ex else '"ㅇㅇ", "ㅎㅇ"'
    off_str = ", ".join(f'"{t}"' for t in offensive_ex) if offensive_ex else '"ㅅㅂ", "존나"'

    prompt = f"""당신은 한국어 채팅 욕설 탐지 전문가입니다.
아래 예시를 참고해 판단하세요.

[위반 아님 - none]: {none_str}
[경고 수준 - offensive]: {off_str}
[즉시 차단 - hate]: 특정 대상 혐오, 심한 욕설 조합

메시지: "{content}"

JSON만 응답, 다른 텍스트 금지:
{{"violation": true/false, "severe": true/false, "reason": "한 줄 이유"}}"""

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
                "stage": 3,
                "score": None,
            }
    except Exception:
        return {"violation": False, "severe": False, "reason": "", "stage": 3, "score": None}


# ── DB 제재 함수들 ────────────────────────────────────────────

async def delete_message_from_redis(party_id: str, content: str) -> bool:
    key = redis_msg_key(party_id)
    messages = await redis_client.lrange(key, 0, -1)
    for raw in reversed(messages):
        try:
            parsed = json.loads(raw)
            if parsed.get("content") == content and parsed.get("type") == "message":
                await redis_client.lrem(key, -1, raw)
                return True
        except Exception:
            continue
    return False


async def delete_message_from_db(party_id: str, user_id: str, content: str):
    try:
        sender_uuid = uuid.UUID(user_id)
        party_uuid = uuid.UUID(party_id)
    except (ValueError, TypeError):
        return
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(PartyChat)
                .where(
                    PartyChat.party_id == party_uuid,
                    PartyChat.sender_id == sender_uuid,
                    PartyChat.message == content,
                    PartyChat.is_deleted == False,
                )
                .order_by(PartyChat.created_at.desc())
                .limit(1)
            )
            chat = result.scalar_one_or_none()
            if chat:
                chat.is_deleted = True
                await db.commit()
    except Exception as e:
        print(f"[DB DELETE ERROR] {e}")


async def _flag_chat_in_db(
    party_id: str,
    user_id: str,
    content: str,
    reason: str,
    moderation_status: str,
    stage: int = 0,
    score: float | None = None,
) -> None:
    try:
        sender_uuid = uuid.UUID(user_id)
        party_uuid = uuid.UUID(party_id)
    except (ValueError, TypeError):
        return
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(PartyChat)
                .where(
                    PartyChat.party_id == party_uuid,
                    PartyChat.sender_id == sender_uuid,
                    PartyChat.message == content,
                )
                .order_by(PartyChat.created_at.desc())
                .limit(1)
            )
            chat = result.scalar_one_or_none()
            if chat:
                chat.is_flagged = True
                chat.flag_reason = reason
                chat.flag_confidence = score
                chat.flag_stage = stage
                chat.moderation_status = moderation_status
                await db.commit()
    except Exception as e:
        print(f"[FLAG DB ERROR] {e}")


async def _ban_user_in_db(party_id: str, user_id: str) -> None:
    try:
        party_uuid = uuid.UUID(party_id)
        user_uuid = uuid.UUID(user_id)
    except (ValueError, TypeError):
        return
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(PartyMember)
                .where(
                    PartyMember.party_id == party_uuid,
                    PartyMember.user_id == user_uuid,
                )
                .values(status="banned")
            )
            party_result = await db.execute(select(Party).where(Party.id == party_uuid))
            party = party_result.scalar_one_or_none()
            if party and party.current_members:
                party.current_members = max(0, party.current_members - 1)
            await db.commit()
    except Exception as e:
        print(f"[BAN DB ERROR] {e}")


async def _apply_trust_penalty(user_id: str, delta: float, reason: str) -> tuple[float, str | None]:
    try:
        from models.mypage.trust_score import TrustScore
        user_uuid = uuid.UUID(user_id)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.id == user_uuid))
            user = result.scalar_one_or_none()
            if not user:
                return 36.5, None
            previous = float(user.trust_score) if user.trust_score is not None else 36.5
            new_score = max(0.0, round(previous + delta, 1))
            user.trust_score = new_score
            row = TrustScore(
                user_id=user_uuid,
                previous_score=previous,
                new_score=new_score,
                change_amount=round(new_score - previous, 1),
                reason=reason,
                created_by=user_uuid,
            )
            db.add(row)
            await db.flush()
            trust_id = str(row.id)
            await db.commit()
            return new_score, trust_id
    except Exception as e:
        print(f"[TRUST PENALTY ERROR] {e}")
        return 36.5, None


async def _increment_chat_warn_count(user_id: str) -> int:
    try:
        user_uuid = uuid.UUID(user_id)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.id == user_uuid))
            user = result.scalar_one_or_none()
            if not user:
                return 0
            user.chat_warn_count = (user.chat_warn_count or 0) + 1
            await db.commit()
            return user.chat_warn_count
    except Exception as e:
        print(f"[WARN COUNT ERROR] {e}")
        return 0


async def _apply_status_by_score(user_id: str, score: float, warn_count: int) -> None:
    try:
        user_uuid = uuid.UUID(user_id)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.id == user_uuid))
            user = result.scalar_one_or_none()
            if not user:
                return
            if score <= 0 or warn_count >= 4:
                user.is_active = False
                user.banned_until = None
            elif score < 10 or warn_count >= 3:
                user.is_active = False
                user.banned_until = datetime.now(timezone.utc) + timedelta(days=30)
            elif score < 20:
                user.banned_until = datetime.now(timezone.utc) + timedelta(hours=72)
            await db.commit()
    except Exception as e:
        print(f"[STATUS BY SCORE ERROR] {e}")


async def _ban_user_ip(user_id: str) -> None:
    try:
        from services.notification_ws_service import notification_connection_manager
        user_uuid = uuid.UUID(user_id)
        async with AsyncSessionLocal() as db:
            ip_result = await db.execute(
                select(RefreshToken.ip_address)
                .where(
                    RefreshToken.user_id == user_uuid,
                    RefreshToken.ip_address != None,
                )
                .order_by(RefreshToken.created_at.desc())
                .limit(1)
            )
            ip = ip_result.scalar_one_or_none()
            if not ip:
                return

            await redis_client.set(f"ip:banned:{ip}", "1")

            token_rows = await db.execute(
                select(RefreshToken.user_id)
                .where(
                    RefreshToken.ip_address == ip,
                    RefreshToken.revoked_at == None,
                    RefreshToken.expires_at > datetime.now(timezone.utc),
                )
                .distinct()
            )
            affected_user_ids = {str(row[0]) for row in token_rows.all()}
            affected_user_ids.add(user_id)

            for uid_str in affected_user_ids:
                try:
                    uid = uuid.UUID(uid_str)
                    await db.execute(
                        RefreshToken.__table__.delete().where(RefreshToken.user_id == uid)
                    )
                    await notification_connection_manager.send_to_user(uid, {
                        "type": "ip_banned",
                        "content": "같은 IP 사용자의 규정 위반으로 접속이 차단되었습니다.",
                    })
                except Exception as e:
                    print(f"[BAN IP USER ERROR] uid={uid_str} {e}")

            await db.commit()
    except Exception as e:
        print(f"[BAN IP ERROR] {e}")


# ── 모더레이션 메인 함수 ─────────────────────────────────────

async def moderate_in_background(party_id: str, user_id: str, content: str, ws: WebSocket):
    moderation = await check_message(content)

    # 닉네임 조회 (시스템 메시지용)
    nickname = "사용자"
    try:
        user_uuid = uuid.UUID(user_id)
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.id == user_uuid))
            u = result.scalar_one_or_none()
            if u and u.nickname:
                nickname = u.nickname
    except Exception:
        pass

    if moderation["severe"]:
        await redis_client.set(blocked_key(user_id), "1", ex=REDIS_TTL)
        await delete_message_from_redis(party_id, content)
        await delete_message_from_db(party_id, user_id, content)
        await _ban_user_in_db(party_id, user_id)
        new_score, trust_ref_id = await _apply_trust_penalty(user_id, -5.0, f"심한 욕설 감지: {moderation['reason']}")
        total_warn = await _increment_chat_warn_count(user_id)
        wk = warn_key(party_id, user_id)
        await redis_client.incr(wk)
        await redis_client.expire(wk, REDIS_TTL)
        await _flag_chat_in_db(
            party_id, user_id, content,
            moderation["reason"], "blocked",
            stage=moderation["stage"], score=moderation["score"],
        )
        await _apply_status_by_score(user_id, new_score, total_warn)
        await _ban_user_ip(user_id)
        await manager.send_personal(ws, {
            "type": "error",
            "content": f"심각한 욕설이 감지되어 차단되었습니다. ({moderation['reason']})",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        await manager.send_personal(ws, {
            "type": "force_logout",
            "ban_type": "trust_score",
            "reference_id": trust_ref_id,
            "content": "심각한 위반으로 계정이 정지되었습니다.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        await manager.broadcast(party_id, {
            "type": "message_deleted",
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        await manager.broadcast(party_id, {
            "type": "system",
            "content": f"{nickname}님이 심각한 욕설로 인해 파티에서 퇴장되었습니다.",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    elif moderation["violation"]:
        wk = warn_key(party_id, user_id)
        party_warn = int(await redis_client.incr(wk))
        await redis_client.expire(wk, REDIS_TTL)
        new_score, trust_ref_id = await _apply_trust_penalty(user_id, -1.0, f"욕설 감지: {moderation['reason']}")
        total_warn = await _increment_chat_warn_count(user_id)
        await _flag_chat_in_db(
            party_id, user_id, content,
            moderation["reason"], "warned",
            stage=moderation["stage"], score=moderation["score"],
        )
        await _apply_status_by_score(user_id, new_score, total_warn)
        if party_warn >= 3:
            await redis_client.set(blocked_key(user_id), "1", ex=REDIS_TTL)
            await _ban_user_in_db(party_id, user_id)
            await _ban_user_ip(user_id)
            await manager.send_personal(ws, {
                "type": "error",
                "content": f"경고 {party_warn}회 누적으로 채팅이 차단되었습니다.",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            await manager.send_personal(ws, {
                "type": "force_logout",
                "ban_type": "trust_score",
                "reference_id": trust_ref_id,
                "content": "경고 누적으로 계정이 정지되었습니다.",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            await manager.broadcast(party_id, {
                "type": "system",
                "content": f"{nickname}님이 경고 누적({party_warn}회)으로 파티에서 퇴장되었습니다.",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            await manager.send_personal(ws, {
                "type": "warning",
                "content": f"경고 {party_warn}/3회: 부적절한 표현이 감지되었습니다. (신뢰도 -1점)",
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
