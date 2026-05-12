from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.quick_match.training_events import (
    QuickMatchTrainingEvent,
    QuickMatchTrainingLabelStatus,
)
from models.quick_match.training_stats import QuickMatchTrainingStat


STAT_TYPE_GLOBAL = "global"
STAT_TYPE_SERVICE = "service"
STAT_TYPE_TRUST_BUCKET = "trust_bucket"
STAT_TYPE_DURATION_MATCH = "duration_match"
STAT_TYPE_CAPACITY_BUCKET = "capacity_bucket"
GLOBAL_STAT_KEY = "all"


def trust_score_to_bucket(value: Any) -> str | None:
    if value in (None, ""):
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if score < 30:
        return "under_30"
    if score < 40:
        return "30_40"
    if score < 50:
        return "40_50"
    if score < 60:
        return "50_60"
    return "over_60"


def capacity_to_bucket(max_members: Any, current_members: Any) -> str | None:
    try:
        max_value = float(max_members or 0)
        current_value = float(current_members or 0)
    except (TypeError, ValueError):
        return None

    if max_value <= 0:
        return "unknown"

    ratio = max(max_value - current_value, 0) / max_value
    if ratio <= 0:
        return "full"
    if ratio <= 0.25:
        return "low"
    if ratio <= 0.50:
        return "medium"
    return "high"


def _normalize_duration_preference(value: Any) -> str | None:
    if value in (None, ""):
        return None

    normalized = str(value).strip().lower().replace(" ", "")
    aliases = {
        "under_1_month": "under_1_month",
        "under1month": "under_1_month",
        "under_1_months": "under_1_month",
        "1개월이하": "under_1_month",
        "1_3_months": "1_3_months",
        "1-3_months": "1_3_months",
        "1~3개월": "1_3_months",
        "1-3개월": "1_3_months",
        "1개월~3개월": "1_3_months",
        "1개월-3개월": "1_3_months",
        "over_3_months": "over_3_months",
        "over3months": "over_3_months",
        "3개월이상": "over_3_months",
        "flexible": "flexible",
    }
    return aliases.get(normalized, normalized)


def _duration_preference_to_range(value: Any) -> tuple[float, float] | None:
    normalized = _normalize_duration_preference(value)
    if normalized in (None, "flexible"):
        return None
    if normalized == "under_1_month":
        return (0.0, 1.0)
    if normalized == "1_3_months":
        return (1.0, 3.0)
    if normalized == "over_3_months":
        return (3.0, float("inf"))
    return None


def duration_match_key(user_value: Any, party_value: Any) -> str:
    normalized_user = _normalize_duration_preference(user_value)
    normalized_party = _normalize_duration_preference(party_value)

    if normalized_user is None:
        return "no_preference"
    if normalized_party is None:
        return "party_unknown"
    if normalized_user == "flexible" or normalized_party == "flexible":
        return "flexible"
    if normalized_user == normalized_party:
        return "exact"

    user_range = _duration_preference_to_range(normalized_user)
    party_range = _duration_preference_to_range(normalized_party)
    if not user_range or not party_range:
        return "unknown"

    user_low, user_high = user_range
    party_low, party_high = party_range
    overlap = max(0.0, min(user_high, party_high) - max(user_low, party_low))
    if overlap > 0:
        return "overlap"
    if user_high == party_low or party_high == user_low:
        return "boundary"
    return "mismatch"


def extract_training_duration_match_key(features_snapshot: dict[str, Any]) -> str:
    request_snapshot = features_snapshot.get("request") or {}
    preferred_conditions = (
        request_snapshot.get("preferred_conditions")
        or features_snapshot.get("preferred_conditions")
        or {}
    )
    party_snapshot = features_snapshot.get("party") or {}
    party_duration = party_snapshot.get("duration_bucket")

    if party_duration is None:
        filter_reasons = features_snapshot.get("filter_reasons") or {}
        hard_filter = filter_reasons.get("hard_filter") or {}
        party_duration = hard_filter.get("party_duration_bucket")

    return duration_match_key(
        preferred_conditions.get("duration_preference"),
        party_duration,
    )


def success_rate(
    bucket: dict[str, int] | None,
    *,
    prior_rate: float,
    prior_weight: int = 20,
) -> tuple[float, int, int]:
    if not bucket:
        return round(prior_rate, 4), 0, 0

    success = int(bucket.get("success", 0) or 0)
    total = int(bucket.get("total", 0) or 0)
    if total <= 0:
        return round(prior_rate, 4), 0, 0

    rate = (success + (prior_rate * prior_weight)) / (total + prior_weight)
    return round(min(max(rate, 0.0), 1.0), 4), success, total


def _increment(stats: dict[str, dict[str, dict[str, int]]], stat_type: str, stat_key: str | None, success: bool) -> None:
    if not stat_key:
        return
    bucket = stats[stat_type].setdefault(stat_key, {"success": 0, "failed": 0, "total": 0})
    bucket["total"] += 1
    if success:
        bucket["success"] += 1
    else:
        bucket["failed"] += 1


async def rebuild_training_stats(db: AsyncSession) -> dict[str, Any]:
    result = await db.execute(
        select(
            QuickMatchTrainingEvent.service_id,
            QuickMatchTrainingEvent.match_success,
            QuickMatchTrainingEvent.features_snapshot,
        ).where(
            QuickMatchTrainingEvent.label_status.in_(
                [
                    QuickMatchTrainingLabelStatus.SUCCESS.value,
                    QuickMatchTrainingLabelStatus.FAILED.value,
                ]
            ),
            QuickMatchTrainingEvent.match_success.isnot(None),
        )
    )
    rows = result.all()

    stats: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for service_id, match_success, features_snapshot in rows:
        success = bool(match_success)
        features = features_snapshot or {}
        user_snapshot = features.get("user") or {}
        party_snapshot = features.get("party") or {}

        _increment(stats, STAT_TYPE_GLOBAL, GLOBAL_STAT_KEY, success)
        _increment(stats, STAT_TYPE_SERVICE, str(service_id), success)
        _increment(
            stats,
            STAT_TYPE_TRUST_BUCKET,
            trust_score_to_bucket(user_snapshot.get("trust_score")),
            success,
        )
        _increment(
            stats,
            STAT_TYPE_DURATION_MATCH,
            extract_training_duration_match_key(features),
            success,
        )
        _increment(
            stats,
            STAT_TYPE_CAPACITY_BUCKET,
            capacity_to_bucket(
                party_snapshot.get("max_members"),
                party_snapshot.get("current_members"),
            ),
            success,
        )

    generated_at = datetime.now(timezone.utc)
    await db.execute(delete(QuickMatchTrainingStat))

    created_count = 0
    for stat_type, buckets in stats.items():
        for stat_key, counts in buckets.items():
            total = int(counts.get("total", 0) or 0)
            success_count = int(counts.get("success", 0) or 0)
            failed_count = int(counts.get("failed", 0) or 0)
            rate = round(success_count / total, 4) if total else 0.0

            db.add(
                QuickMatchTrainingStat(
                    stat_type=stat_type,
                    stat_key=stat_key,
                    success_count=success_count,
                    failed_count=failed_count,
                    total_count=total,
                    success_rate=rate,
                    metadata_snapshot={
                        "source": "quick_match_training_events",
                    },
                    generated_at=generated_at,
                )
            )
            created_count += 1

    await db.flush()
    return {
        "created_count": created_count,
        "event_sample_count": len(rows),
        "generated_at": generated_at.isoformat(),
    }


async def load_training_stats(db: AsyncSession) -> dict[str, Any]:
    result = await db.execute(select(QuickMatchTrainingStat))
    rows = result.scalars().all()

    stats: dict[str, Any] = {
        "global": {"success": 0, "failed": 0, "total": 0},
        "service": {},
        "trust_bucket": {},
        "duration_match": {},
        "capacity_bucket": {},
    }

    for row in rows:
        bucket = {
            "success": int(row.success_count or 0),
            "failed": int(row.failed_count or 0),
            "total": int(row.total_count or 0),
            "success_rate": float(row.success_rate or 0),
        }

        if row.stat_type == STAT_TYPE_GLOBAL and row.stat_key == GLOBAL_STAT_KEY:
            stats["global"] = bucket
        elif row.stat_type in stats and isinstance(stats[row.stat_type], dict):
            stats[row.stat_type][row.stat_key] = bucket

    return stats
