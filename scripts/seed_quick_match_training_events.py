from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.database import AsyncSessionLocal
from models.notification import Notification  # noqa: F401
from models.party import Party, PartyMember, Service  # noqa: F401
from models.payment import Payment  # noqa: F401
from models.quick_match.candidate import (
    QuickMatchCandidate,
    QuickMatchCandidateStatus,
)
from models.quick_match.request import QuickMatchRequest, QuickMatchRequestStatus
from models.quick_match.result import QuickMatchResult  # noqa: F401
from models.quick_match.training_events import (
    QuickMatchTrainingEvent,
    QuickMatchTrainingLabelStatus,
)
from models.report import Report  # noqa: F401
from models.user import User


DEFAULT_COUNT = 500
DEFAULT_SEED = 20260511
SOURCE = "synthetic_seed_v1"

LABEL_WEIGHTS = {
    QuickMatchTrainingLabelStatus.SUCCESS.value: 0.50,
    QuickMatchTrainingLabelStatus.FAILED.value: 0.30,
    QuickMatchTrainingLabelStatus.EXCLUDED.value: 0.15,
    QuickMatchTrainingLabelStatus.PENDING.value: 0.05,
}

FAILED_REASONS = [
    "voluntary_leave_within_3_days",
    "leader_kicked_within_3_days",
    "payment_noshow",
    "payment_missing_after_retention",
    "approved_user_report",
    "chat_moderation_banned",
]

EXCLUDED_REASONS = [
    "not_selected",
    "no_recruiting_party",
    "no_candidate",
    "party_status_changed",
    "approved_party_report",
]

DURATION_BUCKETS = [
    "under_1_month",
    "1_3_months",
    "over_3_months",
    "flexible",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _weighted_label(rng: random.Random) -> str:
    labels = list(LABEL_WEIGHTS.keys())
    weights = list(LABEL_WEIGHTS.values())
    return rng.choices(labels, weights=weights, k=1)[0]


def _duration_bucket(party: Party) -> str | None:
    if not party.start_date or not party.end_date:
        return None

    duration_days = (party.end_date - party.start_date).days + 1
    if duration_days <= 0:
        return None
    if duration_days <= 31:
        return "under_1_month"
    if duration_days <= 93:
        return "1_3_months"
    return "over_3_months"


def _duration_score(user_value: str | None, party_value: str | None) -> float:
    if user_value is None:
        return 0.7
    if party_value is None:
        return 0.6
    if user_value == "flexible" or party_value == "flexible":
        return 0.8
    if user_value == party_value:
        return 1.0
    return 0.3


def _score_candidate(
    *,
    rng: random.Random,
    user: User,
    party: Party,
    duration_preference: str | None,
    label_status: str,
) -> tuple[float, float, float, dict[str, Any]]:
    user_trust_score = float(getattr(user, "trust_score", 36.5) or 36.5)
    min_trust_score = float(getattr(party, "min_trust_score", 0) or 0)
    party_max_members = float(getattr(party, "max_members", 0) or 0)
    party_current_members = float(getattr(party, "current_members", 0) or 0)
    party_duration = _duration_bucket(party)

    if min_trust_score > 0:
        trust_fit_score = 0.0 if user_trust_score < min_trust_score else min(
            1.0,
            0.7 + (min(user_trust_score - min_trust_score, 20.0) / 20.0) * 0.3,
        )
    else:
        trust_fit_score = (
            max(0.0, user_trust_score / 36.5) * 0.7
            if user_trust_score < 36.5
            else min(1.0, 0.7 + (min(user_trust_score - 36.5, 20.0) / 20.0) * 0.3)
        )

    capacity_score = 0.0
    if party_max_members > 0:
        capacity_score = min(
            1.0,
            max(party_max_members - party_current_members, 0) / party_max_members,
        )

    duration_fit_score = _duration_score(duration_preference, party_duration)
    rule_score = (
        trust_fit_score * 0.45
        + capacity_score * 0.30
        + duration_fit_score * 0.25
    )

    reliability_base = min(1.0, user_trust_score / 100.0)
    label_adjustment = {
        QuickMatchTrainingLabelStatus.SUCCESS.value: rng.uniform(0.05, 0.18),
        QuickMatchTrainingLabelStatus.FAILED.value: rng.uniform(-0.22, -0.06),
        QuickMatchTrainingLabelStatus.EXCLUDED.value: rng.uniform(-0.15, 0.04),
        QuickMatchTrainingLabelStatus.PENDING.value: rng.uniform(-0.05, 0.08),
    }[label_status]
    probability_score = min(max(reliability_base + label_adjustment, 0.0), 1.0)
    final_score = (rule_score * 0.55) + (probability_score * 0.45)

    return (
        round(rule_score, 4),
        round(probability_score, 4),
        round(final_score, 4),
        {
            "trust_fit_score": round(trust_fit_score, 4),
            "capacity_score": round(capacity_score, 4),
            "duration_score": round(duration_fit_score, 4),
            "user_trust_score": user_trust_score,
            "party_min_trust_score": min_trust_score,
            "party_duration_bucket": party_duration,
        },
    )


def _label_timestamps(
    *,
    rng: random.Random,
    label_status: str,
    now: datetime,
) -> tuple[datetime, datetime | None, datetime | None]:
    if label_status == QuickMatchTrainingLabelStatus.PENDING.value:
        joined_at = now - timedelta(days=rng.randint(0, 6), hours=rng.randint(0, 23))
        return joined_at, None, None

    joined_at = now - timedelta(days=rng.randint(31, 180), hours=rng.randint(0, 23))
    labeled_at = joined_at + timedelta(days=rng.randint(30, 45))

    if label_status == QuickMatchTrainingLabelStatus.FAILED.value:
        left_at = joined_at + timedelta(hours=rng.randint(1, 72))
        return joined_at, left_at, min(labeled_at, now)

    return joined_at, None, min(labeled_at, now)


def _result_snapshot(
    *,
    label_status: str,
    label_reason: str | None,
    joined_at: datetime,
    left_at: datetime | None,
    labeled_at: datetime | None,
) -> dict[str, Any]:
    match_success = None
    if label_status == QuickMatchTrainingLabelStatus.SUCCESS.value:
        match_success = True
    elif label_status == QuickMatchTrainingLabelStatus.FAILED.value:
        match_success = False

    return {
        "seed": {
            "source": SOURCE,
            "generated_at": _now().isoformat(),
        },
        "label": {
            "match_success": match_success,
            "label_status": label_status,
            "label_reason": label_reason,
            "labeled_at": labeled_at.isoformat() if labeled_at else None,
        },
        "membership": {
            "joined_at": joined_at.isoformat(),
            "left_at": left_at.isoformat() if left_at else None,
        },
    }


async def _clear_seed_data() -> int:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(QuickMatchTrainingEvent.id).where(
                QuickMatchTrainingEvent.features_snapshot["seed"]["source"].as_string()
                == SOURCE
            )
        )
        event_ids = result.scalars().all()

        request_result = await db.execute(
            select(QuickMatchTrainingEvent.request_id).where(
                QuickMatchTrainingEvent.features_snapshot["seed"]["source"].as_string()
                == SOURCE
            )
        )
        request_ids = request_result.scalars().all()

        if event_ids:
            await db.execute(
                delete(QuickMatchTrainingEvent).where(
                    QuickMatchTrainingEvent.id.in_(event_ids)
                )
            )
        if request_ids:
            await db.execute(
                delete(QuickMatchRequest).where(QuickMatchRequest.id.in_(request_ids))
            )

        await db.commit()
        return len(event_ids)


async def seed_training_events(
    *,
    count: int,
    seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    rng = random.Random(seed)
    now = _now()

    async with AsyncSessionLocal() as db:
        user_result = await db.execute(
            select(User).where(User.is_active.is_(True)).limit(max(count * 3, 100))
        )
        users = user_result.scalars().all()

        party_result = await db.execute(
            select(Party).where(Party.service_id.isnot(None)).limit(max(count * 3, 100))
        )
        parties = party_result.scalars().all()

        if len(users) < 2:
            raise RuntimeError("활성 유저가 부족합니다. users 데이터가 필요합니다.")
        if not parties:
            raise RuntimeError("파티 데이터가 부족합니다. parties 데이터가 필요합니다.")

        parties_by_service: dict[Any, list[Party]] = defaultdict(list)
        for party in parties:
            parties_by_service[party.service_id].append(party)

        created = 0
        label_counts: dict[str, int] = defaultdict(int)

        for idx in range(count):
            party = rng.choice(parties)
            candidate_users = [
                user for user in users if getattr(user, "id", None) != party.leader_id
            ]
            if not candidate_users:
                continue

            user = rng.choice(candidate_users)
            label_status = _weighted_label(rng)
            is_selected = label_status != QuickMatchTrainingLabelStatus.EXCLUDED.value
            is_joined = label_status in {
                QuickMatchTrainingLabelStatus.SUCCESS.value,
                QuickMatchTrainingLabelStatus.FAILED.value,
                QuickMatchTrainingLabelStatus.PENDING.value,
            }

            if label_status == QuickMatchTrainingLabelStatus.SUCCESS.value:
                label_reason = "retained_1_month"
            elif label_status == QuickMatchTrainingLabelStatus.FAILED.value:
                label_reason = rng.choice(FAILED_REASONS)
            elif label_status == QuickMatchTrainingLabelStatus.EXCLUDED.value:
                label_reason = rng.choice(EXCLUDED_REASONS)
            else:
                label_reason = None

            joined_at, left_at, labeled_at = _label_timestamps(
                rng=rng,
                label_status=label_status,
                now=now,
            )
            duration_preference = rng.choice(DURATION_BUCKETS + [None])
            preferred_conditions = (
                {"duration_preference": duration_preference}
                if duration_preference
                else {}
            )

            rule_score, probability_score, final_score, score_detail = _score_candidate(
                rng=rng,
                user=user,
                party=party,
                duration_preference=duration_preference,
                label_status=label_status,
            )

            request = QuickMatchRequest(
                user_id=user.id,
                service_id=party.service_id,
                status=(
                    QuickMatchRequestStatus.MATCHED
                    if is_selected
                    else QuickMatchRequestStatus.FAILED
                ),
                preferred_conditions=preferred_conditions,
                matched_party_id=party.id if is_selected else None,
                fail_reason=label_reason if not is_selected else None,
                request_profile_snapshot={
                    "seed": {"source": SOURCE},
                    "user_id": str(user.id),
                    "service_id": str(party.service_id),
                    "trust_score": float(getattr(user, "trust_score", 36.5) or 36.5),
                    "preferred_conditions": preferred_conditions,
                },
                requested_at=joined_at - timedelta(minutes=rng.randint(3, 30)),
                matched_at=joined_at if is_selected else None,
                expired_at=joined_at + timedelta(minutes=5),
                is_active=False,
            )
            db.add(request)
            await db.flush()

            service_parties = parties_by_service.get(party.service_id) or [party]
            candidate_party = party if is_selected else rng.choice(service_parties)
            candidate = QuickMatchCandidate(
                request_id=request.id,
                party_id=candidate_party.id,
                rule_score=rule_score,
                probability_score=probability_score,
                final_score=final_score,
                rank=1 if is_selected else None,
                status=(
                    QuickMatchCandidateStatus.SELECTED
                    if is_selected
                    else QuickMatchCandidateStatus.REJECTED
                ),
                filter_reasons={
                    "seed": {"source": SOURCE, "index": idx},
                    "score_basis": "rule_probability",
                    "rule_reason": score_detail,
                    "excluded_reason": label_reason if not is_selected else None,
                },
            )
            db.add(candidate)
            await db.flush()

            features_snapshot = {
                "seed": {
                    "source": SOURCE,
                    "index": idx,
                    "random_seed": seed,
                },
                "request": {
                    "preferred_conditions": preferred_conditions,
                },
                "user": {
                    "id": str(user.id),
                    "trust_score": float(getattr(user, "trust_score", 36.5) or 36.5),
                },
                "party": {
                    "id": str(candidate_party.id),
                    "service_id": str(candidate_party.service_id),
                    "max_members": candidate_party.max_members,
                    "current_members": candidate_party.current_members,
                    "min_trust_score": float(
                        getattr(candidate_party, "min_trust_score", 0) or 0
                    ),
                    "duration_bucket": _duration_bucket(candidate_party),
                },
                "scores": {
                    "rule_score": rule_score,
                    "probability_score": probability_score,
                    "final_score": final_score,
                    "detail": score_detail,
                },
            }

            event = QuickMatchTrainingEvent(
                request_id=request.id,
                candidate_id=candidate.id,
                user_id=user.id,
                service_id=party.service_id,
                party_id=candidate_party.id,
                is_selected=is_selected,
                is_joined=is_joined,
                match_success=(
                    True
                    if label_status == QuickMatchTrainingLabelStatus.SUCCESS.value
                    else False
                    if label_status == QuickMatchTrainingLabelStatus.FAILED.value
                    else None
                ),
                label_status=label_status,
                label_reason=label_reason,
                joined_at=joined_at if is_joined else None,
                left_at=left_at,
                labeled_at=labeled_at,
                features_snapshot=features_snapshot,
                result_snapshot=_result_snapshot(
                    label_status=label_status,
                    label_reason=label_reason,
                    joined_at=joined_at,
                    left_at=left_at,
                    labeled_at=labeled_at,
                ),
            )
            db.add(event)

            created += 1
            label_counts[label_status] += 1

            if created % 100 == 0:
                await db.flush()

        if dry_run:
            await db.rollback()
        else:
            await db.commit()

    return {
        "source": SOURCE,
        "requested_count": count,
        "created_count": created,
        "dry_run": dry_run,
        "label_counts": dict(label_counts),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed synthetic quick match training events."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clear-seed", action="store_true")
    args = parser.parse_args()

    if args.clear_seed:
        deleted = await _clear_seed_data()
        print({"deleted_seed_events": deleted, "source": SOURCE})

    result = await seed_training_events(
        count=args.count,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
