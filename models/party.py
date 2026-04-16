import uuid
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, func, text
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
    min_trust_score: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")  
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    host: Mapped["User"] = relationship("User", back_populates="hosted_parties", foreign_keys=[leader_id])
    service: Mapped["Service"] = relationship("Service", back_populates="parties")
    members: Mapped[list["PartyMember"]] = relationship("PartyMember", back_populates="party")
    chats: Mapped[list["PartyChat"]] = relationship("PartyChat", back_populates="party")


class PartyMember(Base):
    __tablename__ = "party_members"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    party_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parties.id"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False, server_default="member")
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="active")
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    left_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 승인 플로우 / 빠른매칭 관련 컬럼
    join_type: Mapped[str | None] = mapped_column(String(20))  # 'apply' | 'match' | 'direct'
    match_request_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    leader_review_status: Mapped[str | None] = mapped_column(String(20))  # 'pending' | 'approved' | 'rejected'
    matched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rejected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

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
    sender_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False, server_default="text")
    is_flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    flag_reason: Mapped[str | None] = mapped_column(String(100))
    flag_confidence: Mapped[float | None] = mapped_column(Float)
    moderation_status: Mapped[str | None] = mapped_column(String(20))
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    party: Mapped["Party"] = relationship("Party", back_populates="chats")
    sender: Mapped["User"] = relationship("User", foreign_keys=[sender_id])


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
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    parties: Mapped[list["Party"]] = relationship("Party", back_populates="service")
