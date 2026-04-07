import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


# 상원: 이 모델은 회원가입 뒤 사용자가 선택한 관심사를 DB 테이블 user_interests와 연결합니다.
class UserInterest(Base):
    # 상원: 실제 PostgreSQL에서 사용할 테이블 이름을 user_interests로 고정합니다.
    __tablename__ = "user_interests"
    # 상원: 같은 사용자가 같은 관심사를 중복 저장하지 못하도록 복합 유니크 제약을 둡니다.
    __table_args__ = (
        UniqueConstraint("user_id", "interest_name", name="uq_user_interest_user_name"),
    )

    # 상원: 각 관심사 행을 식별할 기본 키 UUID 컬럼입니다.
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # 상원: 이 관심사가 어느 사용자 계정에 속하는지 users.id와 연결합니다.
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 상원: 프론트 관심사 화면에서 선택한 서비스 이름 문자열을 그대로 저장합니다.
    interest_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # 상원: 사용자가 고른 순서를 유지하려고 정렬용 번호를 함께 저장합니다.
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 상원: 관심사 저장 시각을 남겨 조회 순서와 기록 추적에 사용합니다.
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
