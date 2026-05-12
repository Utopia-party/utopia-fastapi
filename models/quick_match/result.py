from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class QuickMatchResult(Base):
    __tablename__ = "quick_match_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quick_match_requests.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    selected_party_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("parties.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    selected_candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quick_match_candidates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    request_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="요청 시점 데이터 스냅샷",
    )

    candidate_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="선정 후보 데이터 스냅샷",
    )

    final_scores: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="rule/probability/final 최종 점수 스냅샷",
    )

    decision_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="최종 선정 사유",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    request = relationship("QuickMatchRequest", back_populates="result")
    selected_party = relationship("Party", foreign_keys=[selected_party_id])
    selected_candidate = relationship("QuickMatchCandidate", foreign_keys=[selected_candidate_id])
