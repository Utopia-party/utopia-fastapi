from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, JSON, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class QuickMatchRequestStatus(str, enum.Enum):
    REQUESTED = "requested"
    MATCHED = "matched"
    FAILED = "failed"
    EXPIRED = "expired"


class QuickMatchRequest(Base):
    __tablename__ = "quick_match_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
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

    status: Mapped[QuickMatchRequestStatus] = mapped_column(
        Enum(QuickMatchRequestStatus, name="quick_match_request_status"),
        nullable=False,
        default=QuickMatchRequestStatus.REQUESTED,
        server_default=QuickMatchRequestStatus.REQUESTED.value,
        index=True,
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="후보 재시도 횟수",
    )

    preferred_conditions: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="사용자 선호 조건 JSON",
    )

    matched_party_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("parties.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    fail_reason: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="최종 실패 사유",
    )

    request_profile_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="요청 시점 사용자 프로필/집계 스냅샷",
    )

    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    matched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    expired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="현재 진행 중인 요청 여부",
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

    user = relationship("User", back_populates="quick_match_requests")
    matched_party = relationship("Party", foreign_keys=[matched_party_id])
    candidates = relationship(
        "QuickMatchCandidate",
        back_populates="request",
        cascade="all, delete-orphan",
    )
    result = relationship(
        "QuickMatchResult",
        back_populates="request",
        uselist=False,
        cascade="all, delete-orphan",
    )
