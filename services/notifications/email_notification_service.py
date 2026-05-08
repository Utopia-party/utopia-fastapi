from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from tasks.email_tasks import send_email_task


EMAIL_NOTIFICATION_EVENT_CODES = {
    # 결제/정산 관련
    # 파티장: 파티원 정산 완료
    "SETTLEMENT_MEMBER_COMPLETED",
    # 파티원: 내가 정산 완료
    "SETTLEMENT_COMPLETED",

    # 파티 신청 관련
    # 파티원: 내 파티 신청 승인
    "PARTY_JOIN_APPROVED",
    # 파티장: 파티 신청 도착
    "PARTY_JOIN_REQUESTED",
}

def should_send_email(event_code: str | None) -> bool:
    if not event_code:
        return False

    return event_code in EMAIL_NOTIFICATION_EVENT_CODES

# 이메일 내용 생성
def build_email_content(
    *,
    event_code: str,
    title: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    party_title = None

    if metadata:
        party_title = metadata.get("party_title")

    # 이벤트 코드별 이메일 제목 정의
    subject_map = {
        # 결제/정산 관련
        "SETTLEMENT_MEMBER_COMPLETED": "[Party-Up] 파티원 정산이 완료되었어요",
        "SETTLEMENT_COMPLETED": "[Party-Up] 파티 정산이 완료되었어요",

        # 파티 신청 관련
        "PARTY_JOIN_APPROVED": "[Party-Up] 파티 참여가 승인되었어요",
        "PARTY_JOIN_REQUESTED": "[Party-Up] 새로운 파티 참여 신청이 도착했어요",
    }

    # 제목 생성
    subject = subject_map.get(event_code, f"[Party-Up] {title}")

    body = f"""
안녕하세요, Party-Up입니다.

{message}

{f"파티명: {party_title}" if party_title else ""}

자세한 내용은 Party-Up에서 확인해주세요.

감사합니다.
Party-Up 드림
""".strip()

    return subject, body


async def enqueue_email_notification(
    db: AsyncSession,
    *,
    user_id: UUID,
    event_code: str | None,
    title: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not should_send_email(event_code):
        return

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if user is None:
        return

    email = getattr(user, "email", None)
    if not email:
        return

    subject, body = build_email_content(
        event_code=event_code,
        title=title,
        message=message,
        metadata=metadata,
    )

    send_email_task.delay(
        email=email,
        subject=subject,
        body=body,
    )