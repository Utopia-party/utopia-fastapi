from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models.party import Party, PartyEmbedding
from services.quick_match.embedding_service import EmbeddingService
from services.quick_match.quick_match_service import QuickMatchService

logger = logging.getLogger(__name__)


class PartyEmbeddingService:
    """
    파티 생성/수정 시 호출해서 파티 임베딩을 선계산한다.
    router/parties.py는 건드리지 않고, 다른 서비스 레이어에서 재사용하기 위한 전용 서비스다.
    """

    def __init__(self) -> None:
        self.quick_match_service = QuickMatchService()

    async def sync_party_embedding(
        self,
        db: AsyncSession,
        party_id: uuid.UUID,
    ) -> PartyEmbedding | None:
        result = await db.execute(
            select(Party)
            .options(selectinload(Party.service))
            .where(Party.id == party_id)
        )
        party = result.scalar_one_or_none()
        if not party:
            return None

        party_profile = self.quick_match_service._build_party_profile(party)
        embedding_vector = await EmbeddingService.generate_party_embedding(party_profile)
        if not embedding_vector:
            logger.warning(
                "[PartyEmbeddingSync] empty embedding generated party_id=%s",
                party_id,
            )
            return None

        existing_result = await db.execute(
            select(PartyEmbedding).where(PartyEmbedding.party_id == party.id)
        )
        embedding = existing_result.scalar_one_or_none()

        if embedding:
            embedding.embedding_vector = embedding_vector
            if hasattr(embedding, "source_snapshot"):
                embedding.source_snapshot = party_profile
            if hasattr(embedding, "last_generated_at"):
                embedding.last_generated_at = datetime.now(timezone.utc)
        else:
            embedding = PartyEmbedding(
                party_id=party.id,
                service_id=party.service_id,
                embedding_vector=embedding_vector,
            )
            if hasattr(embedding, "source_snapshot"):
                embedding.source_snapshot = party_profile
            if hasattr(embedding, "last_generated_at"):
                embedding.last_generated_at = datetime.now(timezone.utc)
            db.add(embedding)

        await db.flush()

        logger.info(
            "[PartyEmbeddingSync] sync done party_id=%s service_id=%s",
            party.id,
            party.service_id,
        )
        return embedding
