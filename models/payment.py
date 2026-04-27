import uuid
from datetime import datetime
from sqlalchemy import Integer, String, DateTime, Float, ForeignKey, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base


class Payment(Base):
    __tablename__ = "payments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    party_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("parties.id"), nullable=False
    )
    base_price: Mapped[int] = mapped_column(Integer, nullable=False)
    commission_rate: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.30")
    commission_amount: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    discount_reason: Mapped[str | None] = mapped_column(String(50))
    amount: Mapped[int] = mapped_column(Integer, nullable=False)
    payment_method: Mapped[str | None] = mapped_column(String(30))  # 'card' | 'transfer'
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending", index=True)
    # 'pending' | 'approved' | 'rejected'
    billing_month: Mapped[str] = mapped_column(String(7), nullable=False)  # '2026-04'
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    pricing_type: Mapped[str | None] = mapped_column(String(20))  # 'quick_match' | 'normal'
    pg_provider: Mapped[str | None] = mapped_column(String(30))   # 'portone'
    pg_transaction_id: Mapped[str | None] = mapped_column(String(100))

    user: Mapped["User"] = relationship("User")
    party: Mapped["Party"] = relationship("Party")
