from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.party import Party
from services.notification_service import notify_user


PARTY_REFERENCE_TYPE = "party"


def _party_title(party: Party) -> str:
    return party.title or "파티"


def _base_party_metadata(
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


# 파티장
async def notify_party_join_requested(
    db: AsyncSession,
    *,
    party: Party,
    applicant_user_id: UUID,
    applicant_nickname: str | None = None,
) -> None:
    """
    파티장용: 일반 참여 신청 도착
    """
    applicant_name = applicant_nickname or "새 파티원"
    await notify_user(
        db=db,
        user_id=party.leader_id,
        type="party",
        title="새로운 참여 신청이 도착했어요",
        message=f"{applicant_name}님이 [{_party_title(party)}] 파티 참여를 신청했어요.",
        reference_type=PARTY_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_party_metadata(
            event_code="PARTY_JOIN_REQUESTED",
            party=party,
            extra={
                "applicant_user_id": str(applicant_user_id),
                "applicant_nickname": applicant_nickname,
            },
        ),
    )


async def notify_quick_match_member_joined_to_leader(
    db: AsyncSession,
    *,
    party: Party,
    member_user_id: UUID,
    member_nickname: str | None = None,
    match_request_id: UUID | None = None,
) -> None:
    """
    파티장용: 빠른매칭으로 팀원 합류
    """
    member_name = member_nickname or "새 파티원"
    await notify_user(
        db=db,
        user_id=party.leader_id,
        type="party",
        title="빠른매칭으로 새 파티원이 합류했어요",
        message=f"{member_name}님이 빠른매칭을 통해 [{_party_title(party)}] 파티에 합류했어요.",
        reference_type=PARTY_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_party_metadata(
            event_code="PARTY_QUICK_MATCH_JOINED",
            party=party,
            extra={
                "member_user_id": str(member_user_id),
                "member_nickname": member_nickname,
                "join_type": "quick_match",
                "match_request_id": str(match_request_id) if match_request_id else None,
            },
        ),
    )


async def notify_party_member_joined_to_leader(
    db: AsyncSession,
    *,
    party: Party,
    member_user_id: UUID,
    member_nickname: str | None = None,
    join_type: str | None = None,
) -> None:
    """
    파티장용: 파티원 입장 완료
    일반 승인 입장 / 즉시 입장 / 기타 공통
    """
    member_name = member_nickname or "새 파티원"
    await notify_user(
        db=db,
        user_id=party.leader_id,
        type="party",
        title="새 파티원이 입장했어요",
        message=f"{member_name}님이 [{_party_title(party)}] 파티에 합류했어요.",
        reference_type=PARTY_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_party_metadata(
            event_code="PARTY_MEMBER_JOINED",
            party=party,
            extra={
                "member_user_id": str(member_user_id),
                "member_nickname": member_nickname,
                "join_type": join_type,
            },
        ),
    )


# 파티원
async def notify_party_join_request_submitted(
    db: AsyncSession,
    *,
    party: Party,
    applicant_user_id: UUID,
) -> None:
    """
    파티원용: 참여 신청 완료
    """
    await notify_user(
        db=db,
        user_id=applicant_user_id,
        type="party",
        title="파티 참여 신청이 접수되었어요",
        message=f"[{_party_title(party)}] 파티 참여 신청이 완료되었어요. 파티장의 승인을 기다려주세요.",
        reference_type=PARTY_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_party_metadata(
            event_code="PARTY_JOIN_REQUEST_SUBMITTED",
            party=party,
            extra={
                "applicant_user_id": str(applicant_user_id),
            },
        ),
    )


async def notify_party_join_request_rejected(
    db: AsyncSession,
    *,
    party: Party,
    applicant_user_id: UUID,
    reject_reason: str | None = None,
) -> None:
    """
    파티원용: 참여 신청 기각
    """
    message = f"[{_party_title(party)}] 파티 참여 신청이 승인되지 않았어요."
    if reject_reason:
        message += f"\n사유: {reject_reason}"

    await notify_user(
        db=db,
        user_id=applicant_user_id,
        type="party",
        title="파티 참여 신청이 승인되지 않았어요",
        message=message,
        reference_type=PARTY_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_party_metadata(
            event_code="PARTY_JOIN_REQUEST_REJECTED",
            party=party,
            extra={
                "applicant_user_id": str(applicant_user_id),
                "reject_reason": reject_reason,
            },
        ),
    )


async def notify_party_join_approved(
    db: AsyncSession,
    *,
    party: Party,
    member_user_id: UUID,
) -> None:
    """
    파티원용: 파티 참여신청 승인
    """
    await notify_user(
        db=db,
        user_id=member_user_id,
        type="party",
        title="파티 참여가 승인되었어요",
        message=f"[{_party_title(party)}] 파티 참여가 승인되었어요. 이제 파티에 참여할 수 있어요.",
        reference_type=PARTY_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_party_metadata(
            event_code="PARTY_JOIN_APPROVED",
            party=party,
            extra={
                "member_user_id": str(member_user_id),
            },
        ),
    )


async def notify_party_member_kicked(
    db: AsyncSession,
    *,
    party: Party,
    target_user_id: UUID,
    reason: str | None = None,
) -> None:
    """
    파티원용: 강퇴 알림
    """
    message = f"[{_party_title(party)}] 파티 이용이 종료되었어요."
    if reason:
        message += f"\n사유: {reason}"

    await notify_user(
        db=db,
        user_id=target_user_id,
        type="party",
        title="파티 이용이 종료되었어요",
        message=message,
        reference_type=PARTY_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_party_metadata(
            event_code="PARTY_MEMBER_KICKED",
            party=party,
            extra={
                "target_user_id": str(target_user_id),
                "reason": reason,
            },
        ),
    )


async def notify_quick_match_completed(
    db: AsyncSession,
    *,
    party: Party,
    member_user_id: UUID,
    match_request_id: UUID | None = None,
) -> None:
    """
    파티원용: 빠른매칭 완료
    """
    await notify_user(
        db=db,
        user_id=member_user_id,
        type="party",
        title="빠른매칭이 완료되었어요",
        message=f"[{_party_title(party)}] 파티 매칭이 완료되었어요. 이제 파티에 참여할 수 있어요.",
        reference_type=PARTY_REFERENCE_TYPE,
        reference_id=party.id,
        metadata=_base_party_metadata(
            event_code="PARTY_QUICK_MATCH_COMPLETED",
            party=party,
            extra={
                "member_user_id": str(member_user_id),
                "join_type": "quick_match",
                "match_request_id": str(match_request_id) if match_request_id else None,
            },
        ),
    )