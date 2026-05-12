from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class QuickMatchTrainingLabelStatus(str, enum.Enum):
    PENDING = "pending" # 라벨 확정 전
    SUCCESS = "success" # 성공
    FAILED = "failed" # 실패
    EXCLUDED = "excluded" # 학습 제외


class QuickMatchTrainingEvent(Base):
    __tablename__ = "quick_match_training_events"

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

    candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("quick_match_candidates.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
        index=True,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    service_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("services.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    party_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("parties.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    is_selected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    is_joined: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    match_success: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        index=True,
    )

    label_status: Mapped[QuickMatchTrainingLabelStatus] = mapped_column(
        String(20),
        nullable=False,
        default=QuickMatchTrainingLabelStatus.PENDING.value,
        server_default=QuickMatchTrainingLabelStatus.PENDING.value,
        index=True,
    )

    label_reason: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    joined_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    left_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    labeled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    features_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="매칭 후보 생성 시점의 학습 feature snapshot",
    )

    result_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="가입/결제/탈퇴/신고 등 라벨 산정 결과 snapshot",
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

    request = relationship("QuickMatchRequest")
    candidate = relationship("QuickMatchCandidate")
    user = relationship("User")
    party = relationship("Party")
    service = relationship("Service")
