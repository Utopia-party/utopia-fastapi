from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, Numeric, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class QuickMatchCandidateStatus(str, enum.Enum):
    PENDING = "pending"
    SELECTED = "selected"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    FAILED = "failed"


class QuickMatchCandidate(Base):
    __tablename__ = "quick_match_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quick_match_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    party_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("parties.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rule_score: Mapped[float] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0,
        server_default="0",
        comment="규칙 기반 점수",
    )

    probability_score: Mapped[float] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0,
        server_default="0",
        comment="과거 데이터 기반 성공 확률 점수",
    )

    final_score: Mapped[float] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        default=0,
        server_default="0",
        comment="최종 랭킹 점수",
    )

    rank: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="최종 정렬 순위",
    )

    filter_reasons: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="필터링 및 점수 계산 근거",
    )

    status: Mapped[QuickMatchCandidateStatus] = mapped_column(
        Enum(QuickMatchCandidateStatus, name="quick_match_candidate_status"),
        nullable=False,
        default=QuickMatchCandidateStatus.PENDING,
        server_default=QuickMatchCandidateStatus.PENDING.value,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    request = relationship("QuickMatchRequest", back_populates="candidates")
    party = relationship("Party")
