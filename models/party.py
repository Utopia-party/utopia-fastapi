import uuid
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, Index, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base
from datetime import date, datetime

class Party(Base):
    __tablename__ = "parties"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    leader_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    service_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("services.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    max_members: Mapped[int | None] = mapped_column(Integer)
    current_members: Mapped[int | None] = mapped_column(Integer, server_default="1")
    monthly_per_person: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="recruiting")
    # 파티 인원 충족 시 결제 마감일 (3일), 인원 미달 시 NULL
    payment_deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    min_trust_score: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    host: Mapped["User"] = relationship("User", back_populates="hosted_parties", foreign_keys=[leader_id])
    service: Mapped["Service"] = relationship("Service", back_populates="parties")
    members: Mapped[list["PartyMember"]] = relationship("PartyMember", back_populates="party")
    chats: Mapped[list["PartyChat"]] = relationship("PartyChat", back_populates="party")


import uuid
from datetime import datetime

from sqlalchemy import String, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func


from core.database import Base


class PartyMember(Base):
    __tablename__ = "party_members"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    party_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("parties.id"),
        nullable=False,
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="member",
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="active",
    )

    joined_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default=func.now(),
    )

    left_at: Mapped[datetime | None] = mapped_column()

    join_type: Mapped[str | None] = mapped_column(String(20))
    match_request_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    leader_review_status: Mapped[str | None] = mapped_column(String(20))

    matched_at: Mapped[datetime | None] = mapped_column()
    approved_at: Mapped[datetime | None] = mapped_column()
    rejected_at: Mapped[datetime | None] = mapped_column()

    # 관계
    party: Mapped["Party"] = relationship("Party", back_populates="members")
    user: Mapped["User"] = relationship("User", back_populates="party_members")

class PartyChat(Base):
    __tablename__ = "party_chats"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    party_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parties.id"), nullable=False
    )
    sender_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False, server_default="text")
    is_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    flag_reason: Mapped[str | None] = mapped_column(String(100))
    flag_confidence: Mapped[float | None] = mapped_column(Float)
    flag_stage: Mapped[int | None] = mapped_column(Integer)
    moderation_status: Mapped[str | None] = mapped_column(String(20))
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    party: Mapped["Party"] = relationship("Party", back_populates="chats")
    sender: Mapped["User"] = relationship("User", foreign_keys=[sender_id])


class ChatReadStatus(Base):
    __tablename__ = "chat_read_status"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    chat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("party_chats.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    read_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("chat_id", "user_id", name="chat_read_status_chat_id_user_id_key"),
        Index("idx_chat_read_status_chat_id", "chat_id"),
    )


class Service(Base):
    __tablename__ = "services"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    max_members: Mapped[int] = mapped_column(Integer, nullable=False)
    monthly_price: Mapped[int] = mapped_column(Integer, nullable=False)
    original_price: Mapped[int | None] = mapped_column(Integer)
    logo_image_key: Mapped[str | None] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    commission_rate: Mapped[float | None] = mapped_column(Float)
    leader_discount_rate: Mapped[float | None] = mapped_column(Float)
    referral_discount_rate: Mapped[float | None] = mapped_column(Float)
    quick_match_fee_rate: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.05")
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    parties: Mapped[list["Party"]] = relationship("Party", back_populates="service")

from sqlalchemy import JSON


class PartyNotice(Base):
    """채팅방 상단 공지 (방장만 작성/수정/삭제 가능)"""
    __tablename__ = "party_notices"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    party_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parties.id"), nullable=False, unique=True
    )
    created_by: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    party: Mapped["Party"] = relationship("Party")
    author: Mapped["User"] = relationship("User", foreign_keys=[created_by])


class PartyEmbedding(Base):
    __tablename__ = "party_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )

    party_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("parties.id"),
        unique=True,
        nullable=False,
    )

    service_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )

    embedding_vector: Mapped[list[float] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
