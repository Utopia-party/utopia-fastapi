from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class QuickMatchTrainingStat(Base):
    __tablename__ = "quick_match_training_stats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    stat_type: Mapped[str] = mapped_column(String(50), nullable=False)
    stat_key: Mapped[str] = mapped_column(String(255), nullable=False)

    success_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    success_rate: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")

    metadata_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
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

    __table_args__ = (
        UniqueConstraint("stat_type", "stat_key", name="uq_quick_match_training_stats_type_key"),
        Index("idx_quick_match_training_stats_type", "stat_type"),
    )
