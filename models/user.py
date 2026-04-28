import uuid
from sqlalchemy import String, DateTime, Boolean, Numeric, ForeignKey, Integer, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, CheckConstraint, UniqueConstraint, func, text


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(50), nullable=True)
    nickname: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    profile_image_key: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str] = mapped_column(String(50), nullable=False)
    provider: Mapped[str] = mapped_column(String(30), nullable=False, server_default="local")
    provider_id: Mapped[str | None] = mapped_column(String(255))

    role: Mapped[str] = mapped_column(String(20), nullable=False, server_default="USER")
    trust_score: Mapped[float] = mapped_column(Numeric, nullable=False, server_default="36.5")
    chat_warn_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    referrer_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    referrer_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    banned_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    hosted_parties: Mapped[list["Party"]] = relationship("Party", back_populates="host")  # noqa
    party_members: Mapped[list["PartyMember"]] = relationship("PartyMember", back_populates="user")  # noqa
    notifications: Mapped[list["Notification"]] = relationship(
        "Notification",
        foreign_keys="Notification.user_id",
        back_populates="user"
    )  # noqa

    quick_match_requests: Mapped[list["QuickMatchRequest"]] = relationship(
        "QuickMatchRequest",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    referrers: Mapped[list["UserReferrer"]] = relationship(
        "UserReferrer",
        foreign_keys="UserReferrer.user_id",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    referred_users: Mapped[list["UserReferrer"]] = relationship(
        "UserReferrer",
        foreign_keys="UserReferrer.referrer_id",
        back_populates="referrer",
        cascade="all, delete-orphan",
    )

# 추천인     
class UserReferrer(Base):
    __tablename__ = "user_referrers"

    __table_args__ = (
        CheckConstraint("user_id <> referrer_id", name="chk_no_self_referral"),
        UniqueConstraint("user_id", "referrer_id", name="unique_user_referrer"),
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

    referrer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped["User"] = relationship(
        "User",
        foreign_keys=[user_id],
        back_populates="referrers",
    )

    referrer: Mapped["User"] = relationship(
        "User",
        foreign_keys=[referrer_id],
        back_populates="referred_users",
    )