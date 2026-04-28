from sqlalchemy import Column, DateTime, ForeignKey, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID

from core.database import Base


class UserPraise(Base):
    __tablename__ = "user_praises"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )

    from_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )

    to_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )

    praise_type = Column(String(30), nullable=False)

    message = Column(Text, nullable=True)

    created_at = Column(
        DateTime,
        nullable=False,
        server_default=func.now(),
    )

    hidden_from_sender_at = Column(
        DateTime,
        nullable=True,
    )

    hidden_from_receiver_at = Column(
        DateTime,
        nullable=True,
    )