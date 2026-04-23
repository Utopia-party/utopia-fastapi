import json
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel
from typing import Optional
from core.database import get_db, AsyncSessionLocal
from core.config import settings
from models.party import PartyChat
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
async def get_moderation_config():
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
async def update_moderation_config(body: ConfigUpdate):
    config = await get_config()
    update_data = body.model_dump(exclude_none=True)
    config.update(update_data)
    await save_config(config)
    return config


@router.post("/config/reset")
async def reset_moderation_config():
    await save_config(DEFAULT_CONFIG.copy())
    return DEFAULT_CONFIG


# ── 단어 관리 ──

class WordBody(BaseModel):
    word: str


@router.post("/whitelist")
async def add_whitelist(body: WordBody):
    config = await get_config()
    if body.word not in config["whitelist"]:
        config["whitelist"].append(body.word)
        await save_config(config)
    return config["whitelist"]


@router.delete("/whitelist/{word}")
async def remove_whitelist(word: str):
    config = await get_config()
    config["whitelist"] = [w for w in config["whitelist"] if w != word]
    await save_config(config)
    return config["whitelist"]


@router.post("/blacklist")
async def add_blacklist(body: WordBody):
    config = await get_config()
    if body.word not in config["blacklist"]:
        config["blacklist"].append(body.word)
        await save_config(config)
    return config["blacklist"]


@router.delete("/blacklist/{word}")
async def remove_blacklist(word: str):
    config = await get_config()
    config["blacklist"] = [w for w in config["blacklist"] if w != word]
    await save_config(config)
    return config["blacklist"]


# ── 파인튜닝 데이터 현황 ──

@router.get("/finetune/stats")
async def get_finetune_stats(db: AsyncSession = Depends(get_db)):
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
