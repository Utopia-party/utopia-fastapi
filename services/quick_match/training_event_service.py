from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.payment import Payment
from models.party import PartyMember
from models.quick_match.training_events import (
    QuickMatchTrainingEvent,
    QuickMatchTrainingLabelStatus,
)
from models.report import Report


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _merge_snapshot(
    current: dict[str, Any] | None,
    patch: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if patch is None:
        return current

    merged = dict(current or {})
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_snapshot(merged[key], value)
        else:
            merged[key] = value
    return merged


async def get_training_event(
    db: AsyncSession,
    *,
    request_id: uuid.UUID | None = None,
    candidate_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    party_id: uuid.UUID | None = None,
) -> QuickMatchTrainingEvent | None:
    stmt = select(QuickMatchTrainingEvent)

    if candidate_id is not None:
        stmt = stmt.where(QuickMatchTrainingEvent.candidate_id == candidate_id)
    if request_id is not None:
        stmt = stmt.where(QuickMatchTrainingEvent.request_id == request_id)
    if user_id is not None:
        stmt = stmt.where(QuickMatchTrainingEvent.user_id == user_id)
    if party_id is not None:
        stmt = stmt.where(QuickMatchTrainingEvent.party_id == party_id)

    stmt = stmt.order_by(QuickMatchTrainingEvent.created_at.desc()).limit(1)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def mark_training_event_joined(
    db: AsyncSession,
    *,
    request_id: uuid.UUID,
    party_id: uuid.UUID,
    joined_at: datetime | None = None,
    result_snapshot: dict[str, Any] | None = None,
) -> QuickMatchTrainingEvent | None:
    event = await get_training_event(
        db,
        request_id=request_id,
        party_id=party_id,
    )
    if not event:
        return None

    joined_at = joined_at or _now()

    event.is_joined = True
    event.joined_at = joined_at
    event.result_snapshot = _merge_snapshot(
        event.result_snapshot,
        {
            "join": {
                "is_joined": True,
                "joined_at": joined_at.isoformat(),
            },
            **(result_snapshot or {}),
        },
    )
    return event


async def mark_training_event_failed(
    db: AsyncSession,
    *,
    reason: str,
    request_id: uuid.UUID | None = None,
    candidate_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    party_id: uuid.UUID | None = None,
    left_at: datetime | None = None,
    result_snapshot: dict[str, Any] | None = None,
) -> QuickMatchTrainingEvent | None:
    event = await get_training_event(
        db,
        request_id=request_id,
        candidate_id=candidate_id,
        user_id=user_id,
        party_id=party_id,
    )
    if not event:
        return None

    labeled_at = _now()

    event.match_success = False
    event.label_status = QuickMatchTrainingLabelStatus.FAILED.value
    event.label_reason = reason
    event.labeled_at = labeled_at

    if left_at is not None:
        event.left_at = left_at

    event.result_snapshot = _merge_snapshot(
        event.result_snapshot,
        {
            "label": {
                "match_success": False,
                "label_status": QuickMatchTrainingLabelStatus.FAILED.value,
                "label_reason": reason,
                "labeled_at": labeled_at.isoformat(),
            },
            **(result_snapshot or {}),
        },
    )
    return event


async def mark_training_event_excluded(
    db: AsyncSession,
    *,
    reason: str,
    request_id: uuid.UUID | None = None,
    candidate_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    party_id: uuid.UUID | None = None,
    result_snapshot: dict[str, Any] | None = None,
) -> QuickMatchTrainingEvent | None:
    event = await get_training_event(
        db,
        request_id=request_id,
        candidate_id=candidate_id,
        user_id=user_id,
        party_id=party_id,
    )
    if not event:
        return None

    labeled_at = _now()

    event.match_success = None
    event.label_status = QuickMatchTrainingLabelStatus.EXCLUDED.value
    event.label_reason = reason
    event.labeled_at = labeled_at
    event.result_snapshot = _merge_snapshot(
        event.result_snapshot,
        {
            "label": {
                "match_success": None,
                "label_status": QuickMatchTrainingLabelStatus.EXCLUDED.value,
                "label_reason": reason,
                "labeled_at": labeled_at.isoformat(),
            },
            **(result_snapshot or {}),
        },
    )
    return event


async def mark_unselected_candidates_excluded(
    db: AsyncSession,
    *,
    request_id: uuid.UUID,
    reason: str = "not_selected",
) -> int:
    result = await db.execute(
        select(QuickMatchTrainingEvent).where(
            QuickMatchTrainingEvent.request_id == request_id,
            QuickMatchTrainingEvent.is_selected.is_(False),
            QuickMatchTrainingEvent.label_status
            == QuickMatchTrainingLabelStatus.PENDING.value,
        )
    )
    events = result.scalars().all()
    labeled_at = _now()

    for event in events:
        event.match_success = None
        event.label_status = QuickMatchTrainingLabelStatus.EXCLUDED.value
        event.label_reason = reason
        event.labeled_at = labeled_at
        event.result_snapshot = _merge_snapshot(
            event.result_snapshot,
            {
                "label": {
                    "match_success": None,
                    "label_status": QuickMatchTrainingLabelStatus.EXCLUDED.value,
                    "label_reason": reason,
                    "labeled_at": labeled_at.isoformat(),
                }
            },
        )

    return len(events)

async def mark_payment_approved(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    party_id: uuid.UUID,
    payment_id: uuid.UUID | None = None,
    billing_month: str | None = None,
    paid_at: datetime | None = None,
) -> QuickMatchTrainingEvent | None:
    event = await get_training_event(
        db,
        user_id=user_id,
        party_id=party_id,
    )
    if not event:
        return None

    event.result_snapshot = _merge_snapshot(
        event.result_snapshot,
        {
            "payment": {
                "status": "approved",
                "payment_id": str(payment_id) if payment_id else None,
                "billing_month": billing_month,
                "paid_at": paid_at.isoformat() if paid_at else None,
            }
        },
    )
    return event

async def mark_payment_failed(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    party_id: uuid.UUID,
    reason: str = "payment_failed",
    payment_id: uuid.UUID | None = None,
    billing_month: str | None = None,
    failed_at: datetime | None = None,
) -> QuickMatchTrainingEvent | None:
    return await mark_training_event_failed(
        db,
        user_id=user_id,
        party_id=party_id,
        reason=reason,
        result_snapshot={
            "payment": {
                "status": "failed",
                "payment_id": str(payment_id) if payment_id else None,
                "billing_month": billing_month,
                "failed_at": (failed_at or _now()).isoformat(),
            }
        },
    )


async def mark_member_left_failed(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    party_id: uuid.UUID,
    member_status: str,
    left_at: datetime | None = None,
    reason: str | None = None,
) -> QuickMatchTrainingEvent | None:
    left_at = left_at or _now()
    label_reason = reason or f"member_{member_status}"

    return await mark_training_event_failed(
        db,
        user_id=user_id,
        party_id=party_id,
        reason=label_reason,
        left_at=left_at,
        result_snapshot={
            "membership": {
                "status": member_status,
                "left_at": left_at.isoformat(),
            }
        },
    )


async def mark_member_departure_failed_if_within_days(
    db: AsyncSession,
    *,
    user_id: uuid.UUID,
    party_id: uuid.UUID,
    member_status: str,
    left_at: datetime | None = None,
    reason: str | None = None,
    window_days: int = 3,
) -> QuickMatchTrainingEvent | None:
    event = await get_training_event(
        db,
        user_id=user_id,
        party_id=party_id,
    )
    if not event:
        return None

    left_at = left_at or _now()
    joined_at = event.joined_at
    if joined_at and joined_at.tzinfo is None:
        joined_at = joined_at.replace(tzinfo=timezone.utc)
    if left_at.tzinfo is None:
        left_at = left_at.replace(tzinfo=timezone.utc)

    if joined_at and left_at - joined_at > timedelta(days=window_days):
        event.result_snapshot = _merge_snapshot(
            event.result_snapshot,
            {
                "membership": {
                    "status": member_status,
                    "left_at": left_at.isoformat(),
                    "departure_within_label_window": False,
                    "label_window_days": window_days,
                }
            },
        )
        return event

    return await mark_member_left_failed(
        db,
        user_id=user_id,
        party_id=party_id,
        member_status=member_status,
        left_at=left_at,
        reason=reason or f"member_{member_status}_within_{window_days}_days",
    )


async def mark_approved_report_failed(
    db: AsyncSession,
    *,
    report: Report,
) -> int:
    target_type = str(report.target_type or "").lower()
    labeled_count = 0

    if target_type == "user":
        result = await db.execute(
            select(QuickMatchTrainingEvent).where(
                QuickMatchTrainingEvent.user_id == report.target_id,
                QuickMatchTrainingEvent.is_joined.is_(True),
                QuickMatchTrainingEvent.label_status
                == QuickMatchTrainingLabelStatus.PENDING.value,
            )
        )
        events = result.scalars().all()

        for event in events:
            await mark_training_event_failed(
                db,
                request_id=event.request_id,
                party_id=event.party_id,
                reason="approved_user_report",
                result_snapshot={
                    "report": {
                        "report_id": str(report.id),
                        "target_type": report.target_type,
                        "target_id": str(report.target_id),
                        "category": report.category,
                        "status": report.status,
                        "reviewed_at": report.reviewed_at.isoformat()
                        if report.reviewed_at
                        else None,
                    }
                },
            )
            labeled_count += 1

    elif target_type == "party":
        result = await db.execute(
            select(QuickMatchTrainingEvent).where(
                QuickMatchTrainingEvent.party_id == report.target_id,
                QuickMatchTrainingEvent.label_status
                == QuickMatchTrainingLabelStatus.PENDING.value,
            )
        )
        events = result.scalars().all()

        for event in events:
            await mark_training_event_excluded(
                db,
                request_id=event.request_id,
                party_id=event.party_id,
                reason="approved_party_report",
                result_snapshot={
                    "report": {
                        "report_id": str(report.id),
                        "target_type": report.target_type,
                        "target_id": str(report.target_id),
                        "category": report.category,
                        "status": report.status,
                        "reviewed_at": report.reviewed_at.isoformat()
                        if report.reviewed_at
                        else None,
                    }
                },
            )
            labeled_count += 1

    return labeled_count


async def label_retained_successes(
    db: AsyncSession,
    *,
    retention_days: int = 30,
) -> dict[str, int]:
    now = _now()
    cutoff = now - timedelta(days=retention_days)

    result = await db.execute(
        select(QuickMatchTrainingEvent).where(
            QuickMatchTrainingEvent.is_selected.is_(True),
            QuickMatchTrainingEvent.is_joined.is_(True),
            QuickMatchTrainingEvent.joined_at.isnot(None),
            QuickMatchTrainingEvent.joined_at <= cutoff,
            QuickMatchTrainingEvent.label_status
            == QuickMatchTrainingLabelStatus.PENDING.value,
        )
    )
    events = result.scalars().all()

    success_count = 0
    failed_count = 0
    pending_count = 0
    excluded_count = 0

    for event in events:
        member_result = await db.execute(
            select(PartyMember).where(
                PartyMember.party_id == event.party_id,
                PartyMember.user_id == event.user_id,
            )
        )
        member = member_result.scalar_one_or_none()

        if not member:
            await mark_training_event_failed(
                db,
                request_id=event.request_id,
                party_id=event.party_id,
                reason="member_missing_after_retention",
            )
            failed_count += 1
            continue

        member_status = str(member.status or "").lower()
        if member_status in {"left", "kicked", "banned"}:
            await mark_training_event_failed(
                db,
                request_id=event.request_id,
                party_id=event.party_id,
                reason=f"member_{member_status}",
                left_at=member.left_at,
                result_snapshot={
                    "membership": {
                        "status": member.status,
                        "left_at": member.left_at.isoformat()
                        if member.left_at
                        else None,
                    }
                },
            )
            failed_count += 1
            continue

        if member_status != "active":
            pending_count += 1
            continue

        payment_result = await db.execute(
            select(Payment).where(
                Payment.party_id == event.party_id,
                Payment.user_id == event.user_id,
                Payment.status == "approved",
            )
        )
        payment = payment_result.scalar_one_or_none()

        if not payment:
            await mark_training_event_failed(
                db,
                request_id=event.request_id,
                party_id=event.party_id,
                reason="payment_missing_after_retention",
            )
            failed_count += 1
            continue

        approved_user_report_result = await db.execute(
            select(Report).where(
                func.lower(Report.target_type) == "user",
                Report.target_id == event.user_id,
                Report.status == "APPROVED",
                Report.created_at >= event.joined_at,
            )
        )
        approved_user_report = approved_user_report_result.scalar_one_or_none()

        if approved_user_report:
            await mark_training_event_failed(
                db,
                request_id=event.request_id,
                party_id=event.party_id,
                reason="approved_user_report",
                result_snapshot={
                    "report": {
                        "report_id": str(approved_user_report.id),
                        "target_type": approved_user_report.target_type,
                        "category": approved_user_report.category,
                        "status": approved_user_report.status,
                    }
                },
            )
            failed_count += 1
            continue

        approved_party_report_result = await db.execute(
            select(Report).where(
                func.lower(Report.target_type) == "party",
                Report.target_id == event.party_id,
                Report.status == "APPROVED",
                Report.created_at >= event.joined_at,
            )
        )
        approved_party_report = approved_party_report_result.scalar_one_or_none()

        if approved_party_report:
            await mark_training_event_excluded(
                db,
                request_id=event.request_id,
                party_id=event.party_id,
                reason="approved_party_report",
                result_snapshot={
                    "report": {
                        "report_id": str(approved_party_report.id),
                        "target_type": approved_party_report.target_type,
                        "category": approved_party_report.category,
                        "status": approved_party_report.status,
                    }
                },
            )
            excluded_count += 1
            continue

        event.match_success = True
        event.label_status = QuickMatchTrainingLabelStatus.SUCCESS.value
        event.label_reason = "retained_1_month"
        event.labeled_at = now
        event.result_snapshot = _merge_snapshot(
            event.result_snapshot,
            {
                "label": {
                    "match_success": True,
                    "label_status": QuickMatchTrainingLabelStatus.SUCCESS.value,
                    "label_reason": "retained_1_month",
                    "labeled_at": now.isoformat(),
                },
                "membership": {
                    "status": member.status,
                    "joined_at": event.joined_at.isoformat()
                    if event.joined_at
                    else None,
                },
                "payment": {
                    "status": payment.status,
                    "payment_id": str(payment.id),
                    "billing_month": payment.billing_month,
                    "paid_at": payment.paid_at.isoformat()
                    if payment.paid_at
                    else None,
                },
            },
        )
        success_count += 1

    return {
        "success": success_count,
        "failed": failed_count,
        "pending": pending_count,
        "excluded": excluded_count,
    }
