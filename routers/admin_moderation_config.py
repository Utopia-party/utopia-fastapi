import json
import uuid
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException
from routers.admin.deps import require_admin_moderation_permission
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from typing import Optional
from core.database import get_db, AsyncSessionLocal
from core.config import settings
from models.party import PartyChat, PartyMember
from models.user import User
import redis.asyncio as aioredis

router = APIRouter(prefix="/admin/moderation", tags=["admin-moderation"])
redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)

CONFIG_KEY = "moderation:config"

DEFAULT_CONFIG = {
    "stage1_enabled": True,
    "stage2_enabled": True,
    "stage3_enabled": True,
    "stage2_pass_threshold": 0.75,
    "stage2_block_threshold": 0.92,
    "ollama_prompt_examples": [
        {"text": "ㅇㅇ", "label": "none"},
        {"text": "ㅎㅇ", "label": "none"},
        {"text": "ㅋㅋ", "label": "none"},
        {"text": "ㅅㅂ", "label": "offensive"},
        {"text": "존나", "label": "offensive"},
    ],
    "whitelist": ["ㅇㅇ", "ㅎㅇ", "ㅋㅋ", "ㅎㅎ", "ㄱㅊ", "ㄴㄴ", "ㅇㅋ", "ㄱㄱ", "ㅂㅂ"],
    "blacklist": ["ㅅㅂ", "ㅈㄹ", "ㅂㅅ", "ㄷㅊ", "ㅁㅊ"],
}


async def get_config() -> dict:
    raw = await redis_client.get(CONFIG_KEY)
    if raw:
        return json.loads(raw)
    return DEFAULT_CONFIG.copy()


async def save_config(config: dict):
    await redis_client.set(CONFIG_KEY, json.dumps(config, ensure_ascii=False))


# ── 설정 조회/저장 ──

@router.get("/config")
async def get_moderation_config(_: object = Depends(require_admin_moderation_permission)):
    return await get_config()


class ConfigUpdate(BaseModel):
    stage1_enabled: Optional[bool] = None
    stage2_enabled: Optional[bool] = None
    stage3_enabled: Optional[bool] = None
    stage2_pass_threshold: Optional[float] = None
    stage2_block_threshold: Optional[float] = None
    ollama_prompt_examples: Optional[list] = None
    whitelist: Optional[list[str]] = None
    blacklist: Optional[list[str]] = None


@router.patch("/config")
async def update_moderation_config(body: ConfigUpdate, _: object = Depends(require_admin_moderation_permission)):
    config = await get_config()
    update_data = body.model_dump(exclude_none=True)
    config.update(update_data)
    await save_config(config)
    return config


@router.post("/config/reset")
async def reset_moderation_config(_: object = Depends(require_admin_moderation_permission)):
    await save_config(DEFAULT_CONFIG.copy())
    return DEFAULT_CONFIG


# ── 단어 관리 ──

class WordBody(BaseModel):
    word: str


@router.post("/whitelist")
async def add_whitelist(body: WordBody, _: object = Depends(require_admin_moderation_permission)):
    config = await get_config()
    if body.word not in config["whitelist"]:
        config["whitelist"].append(body.word)
        await save_config(config)
    return config["whitelist"]


@router.delete("/whitelist/{word}")
async def remove_whitelist(word: str, _: object = Depends(require_admin_moderation_permission)):
    config = await get_config()
    config["whitelist"] = [w for w in config["whitelist"] if w != word]
    await save_config(config)
    return config["whitelist"]


@router.post("/blacklist")
async def add_blacklist(body: WordBody, _: object = Depends(require_admin_moderation_permission)):
    config = await get_config()
    if body.word not in config["blacklist"]:
        config["blacklist"].append(body.word)
        await save_config(config)
    return config["blacklist"]


@router.delete("/blacklist/{word}")
async def remove_blacklist(word: str, _: object = Depends(require_admin_moderation_permission)):
    config = await get_config()
    config["blacklist"] = [w for w in config["blacklist"] if w != word]
    await save_config(config)
    return config["blacklist"]


# ── 파인튜닝 데이터 현황 ──

@router.get("/finetune/stats")
async def get_finetune_stats(db: AsyncSession = Depends(get_db), _: object = Depends(require_admin_moderation_permission)):
    result = await db.execute(
        select(PartyChat.moderation_status, func.count())
        .where(PartyChat.is_flagged == True)
        .group_by(PartyChat.moderation_status)
    )
    rows = result.all()
    counts = {row[0]: row[1] for row in rows}

    total = sum(counts.values())
    hate = counts.get("blocked", 0)
    offensive = counts.get("warned", 0)
    none_label = counts.get("false_positive", 0)
    ready = total >= 500 and min(hate, offensive, none_label) >= 100

    return {
        "total": total,
        "hate": hate,
        "offensive": offensive,
        "none": none_label,
        "ready": ready,
        "min_required": 500,
    }


# ── 채팅 차단 해제 ──

@router.post("/unblock/user/{user_id}")
async def unblock_chat_user(user_id: str, _: object = Depends(require_admin_moderation_permission)):
    """채팅 욕설로 차단된 유저 완전 해제"""
    # 1) Redis: 채팅 블럭 + warn 카운트 키 전부 삭제
    await redis_client.delete(f"blocked:user:{user_id}")
    warn_keys = await redis_client.keys(f"warn:*:{user_id}")
    if warn_keys:
        await redis_client.delete(*warn_keys)

    try:
        from sqlalchemy import update as sa_update
        user_uuid = uuid.UUID(user_id)
        async with AsyncSessionLocal() as db:
            # 2) DB: is_active 복구, banned_until 초기화, chat_warn_count 리셋
            result = await db.execute(select(User).where(User.id == user_uuid))
            user = result.scalar_one_or_none()
            if user:
                user.is_active = True
                user.banned_until = None
                user.chat_warn_count = 0
            # 3) DB: PartyMember banned → active 복구
            await db.execute(
                sa_update(PartyMember)
                .where(PartyMember.user_id == user_uuid, PartyMember.status == "banned")
                .values(status="active")
            )
            await db.commit()
    except Exception as e:
        print(f"[UNBLOCK USER ERROR] {e}")

    return {"unblocked": True, "user_id": user_id}


# ── 채팅 IP 벤 목록 조회 / 해제 ──

@router.get("/chat-bans")
async def list_chat_bans(_: object = Depends(require_admin_moderation_permission)):
    """채팅 욕설로 IP 벤된 목록 (ip:banned:* 키 기준)"""
    keys = await redis_client.keys("ip:banned:*")
    result = []
    for key in keys:
        ip = key.replace("ip:banned:", "")
        ttl = await redis_client.ttl(key)
        result.append({"ip": ip, "ttl": ttl})
    return result


@router.delete("/unblock/ip/{ip}")
async def unblock_ip_ban(ip: str, _: object = Depends(require_admin_moderation_permission)):
    """채팅 IP 벤 해제"""
    await redis_client.delete(f"ip:banned:{ip}")
    return {"unblocked": True, "ip": ip}
