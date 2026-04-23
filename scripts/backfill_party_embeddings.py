from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import AsyncSessionLocal
from models.party import Party
from models.notification import Notification

from services.quick_match.party_embedding_service import PartyEmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def backfill_party_embeddings() -> None:
    sync_service = PartyEmbeddingService()

    async with AsyncSessionLocal() as db:  # type: AsyncSession
        result = await db.execute(select(Party.id))
        party_ids = result.scalars().all()

        logger.info("[PartyEmbeddingBackfill] total_parties=%s", len(party_ids))

        processed = 0
        for party_id in party_ids:
            try:
                await sync_service.sync_party_embedding(db=db, party_id=party_id)
                processed += 1

                if processed % 100 == 0:
                    await db.commit()
                    logger.info("[PartyEmbeddingBackfill] committed=%s", processed)
            except Exception as exc:
                await db.rollback()
                logger.exception(
                    "[PartyEmbeddingBackfill] failed party_id=%s error=%s",
                    party_id,
                    exc,
                )

        await db.commit()
        logger.info("[PartyEmbeddingBackfill] done processed=%s", processed)


if __name__ == "__main__":
    asyncio.run(backfill_party_embeddings())
