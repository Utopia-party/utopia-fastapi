from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.party import Party
from services.notification_service import notify_user


SETTLEMENT_REFERENCE_TYPE = "settlement"


def _party_title(party: Party) -> str:
    return party.title or "파티"


def _base_settlement_metadata(
    *,
    event_code: str,
    party: Party,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "event_code": event_code,
        "party_id": str(party.id),
        "leader_id": str(party.leader_id),
        "service_id": str(party.service_id),
        "party_title": _party_title(party),
        "party_status": party.status,
    }
    if extra:
        metadata.update(extra)
    return metadata


async def notify_settlement_requested_to_member(
    db: AsyncSession,
    *,
    party: Party,
    member_user_id: UUID,
    amount: int | None = None,
) -> None:
    message = f"[{_party_title(party)}] 파티 정산 요청이 도착했어요."
    if amount is not None:
        message += f"\n정산 금액: {amount:,}원"

    await notify_user(
        db=db,
        user_id=member_user_id,
        type="settlement",
        title="파티 정산 요청이 도착했어요",
        message=message,
        reference_type=SETTLEMENT_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_settlement_metadata(
            event_code="SETTLEMENT_REQUESTED",
            party=party,
            extra={
                "member_user_id": str(member_user_id),
                "amount": amount,
            },
        ),
    )


async def notify_party_started_to_member(
    db: AsyncSession,
    *,
    party: Party,
    member_user_id: UUID,
) -> None:
    await notify_user(
        db=db,
        user_id=member_user_id,
        type="settlement",
        title="파티가 시작되었어요",
        message=f"[{_party_title(party)}] 파티가 시작되었어요. 정산 일정을 확인해주세요.",
        reference_type=SETTLEMENT_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_settlement_metadata(
            event_code="PARTY_STARTED",
            party=party,
            extra={
                "member_user_id": str(member_user_id),
            },
        ),
    )


async def notify_member_settlement_completed_to_leader(
    db: AsyncSession,
    *,
    party: Party,
    member_user_id: UUID,
    member_nickname: str | None = None,
    account_required: bool = True,
) -> None:
    member_name = member_nickname or "파티원"
    message = f"{member_name}님의 정산이 완료되었어요."
    if account_required:
        message += "\n정산 계좌 정보를 확인해주세요."

    await notify_user(
        db=db,
        user_id=party.leader_id,
        type="settlement",
        title="파티원 정산이 완료되었어요",
        message=message,
        reference_type=SETTLEMENT_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_settlement_metadata(
            event_code="SETTLEMENT_MEMBER_COMPLETED",
            party=party,
            extra={
                "member_user_id": str(member_user_id),
                "member_nickname": member_nickname,
                "account_required": account_required,
            },
        ),
    )


async def notify_deposit_scheduled_to_leader(
    db: AsyncSession,
    *,
    party: Party,
    amount: int | None = None,
    scheduled_at: str | None = None,
) -> None:
    message = f"[{_party_title(party)}] 파티 입금이 예정되었어요."
    if amount is not None:
        message += f"\n입금 예정 금액: {amount:,}원"
    if scheduled_at:
        message += f"\n입금 예정 시각: {scheduled_at}"

    await notify_user(
        db=db,
        user_id=party.leader_id,
        type="settlement",
        title="파티 입금이 예정되었어요",
        message=message,
        reference_type=SETTLEMENT_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_settlement_metadata(
            event_code="SETTLEMENT_DEPOSIT_SCHEDULED",
            party=party,
            extra={
                "amount": amount,
                "scheduled_at": scheduled_at,
            },
        ),
    )