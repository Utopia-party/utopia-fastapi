from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.party import Party
from models.party import PartyChat


async def resolve_target_snapshot_name(
    db: AsyncSession,
    target_type: str,
    target_id: UUID,
) -> str | None:
    if target_type == "USER":
        result = await db.execute(
            select(User.nickname).where(User.id == target_id)
        )
        return result.scalar_one_or_none()

    if target_type == "PARTY":
        result = await db.execute(
            select(Party.title).where(Party.id == target_id)
        )
        return result.scalar_one_or_none()

    if target_type == "CHAT":
        result = await db.execute(
            select(PartyChat.message).where(PartyChat.id == target_id)
        )
        message = result.scalar_one_or_none()
        return message[:100] if message else None

    return None