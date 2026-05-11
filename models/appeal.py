import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, ForeignKey, func, text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class BanAppeal(Base):
    __tablename__ = "ban_appeals"

    __table_args__ = (
        UniqueConstraint("user_id", "ban_reference_id", name="uq_appeal_per_ban"),
    )

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

    # 어떤 제재인지: ip_ban / trust_score / manual
    ban_type: Mapped[str] = mapped_column(String(30), nullable=False)

    # 관련 ModerationAction.id 또는 TrustScore.id (ip_ban이면 None 가능)
    ban_reference_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    reason: Mapped[str] = mapped_column(Text, nullable=False)

    # 이의제기 신청 시점의 IP
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)

    # PENDING / APPROVED / REJECTED
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="PENDING")

    reviewed_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    admin_memo: Mapped[str | None] = mapped_column(Text, nullable=True)

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

    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])  # noqa
    reviewer: Mapped["User"] = relationship("User", foreign_keys=[reviewed_by])  # noqa
