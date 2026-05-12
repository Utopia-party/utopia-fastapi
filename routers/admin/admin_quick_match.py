from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database import get_db
from models.party import Party, Service
from models.quick_match.candidate import QuickMatchCandidate
from models.quick_match.request import QuickMatchRequest
from models.quick_match.result import QuickMatchResult
from models.quick_match.training_events import QuickMatchTrainingEvent
from models.quick_match.training_stats import QuickMatchTrainingStat
from models.user import User
from routers.admin.deps import AdminContext, require_admin_quick_match_permission
from services.quick_match.quick_match_service import QuickMatchService
from services.quick_match.training_event_service import label_retained_successes
from services.quick_match.training_stats_service import rebuild_training_stats

router = APIRouter(
    prefix="/admin/quick-match",
    tags=["Admin Quick Match"],
)

quick_match_service = QuickMatchService()


def _value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _iso(value: datetime | None) -> str | None:
    if not value:
        return None
    return value.isoformat()


def _as_float(value: Any) -> float:
    return float(value or 0)


def _elapsed_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if not start or not end:
        return None
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    return round(max((end - start).total_seconds(), 0), 2)


def _date_bounds(date_from: date | None, date_to: date | None) -> tuple[datetime | None, datetime | None]:
    start = datetime.combine(date_from, time.min) if date_from else None
    end = datetime.combine(date_to, time.max) if date_to else None
    return start, end


async def _service_name_map(db: AsyncSession, service_ids: set[uuid.UUID]) -> dict[str, str]:
    if not service_ids:
        return {}
    result = await db.execute(select(Service.id, Service.name).where(Service.id.in_(service_ids)))
    return {str(service_id): name for service_id, name in result.all()}


def _party_name(party: Party | None, party_id: uuid.UUID | None = None) -> str | None:
    if party:
        return party.title or str(party.id)
    return str(party_id) if party_id else None


def _candidate_out(candidate: QuickMatchCandidate) -> dict[str, Any]:
    party = getattr(candidate, "party", None)
    return {
        "candidateId": str(candidate.id),
        "requestId": str(candidate.request_id),
        "partyId": str(candidate.party_id),
        "partyName": _party_name(party, candidate.party_id),
        "rank": candidate.rank,
        "status": _value(candidate.status),
        "ruleScore": _as_float(candidate.rule_score),
        "probabilityScore": _as_float(candidate.probability_score),
        "finalScore": _as_float(candidate.final_score),
        "filterReasons": candidate.filter_reasons or {},
        "createdAt": _iso(candidate.created_at),
        "updatedAt": _iso(candidate.updated_at),
    }


def _training_event_out(event: QuickMatchTrainingEvent) -> dict[str, Any]:
    return {
        "eventId": str(event.id),
        "requestId": str(event.request_id),
        "candidateId": str(event.candidate_id) if event.candidate_id else None,
        "userId": str(event.user_id),
        "serviceId": str(event.service_id),
        "partyId": str(event.party_id),
        "isSelected": bool(event.is_selected),
        "isJoined": bool(event.is_joined),
        "matchSuccess": event.match_success,
        "labelStatus": _value(event.label_status),
        "labelReason": event.label_reason,
        "joinedAt": _iso(event.joined_at),
        "leftAt": _iso(event.left_at),
        "labeledAt": _iso(event.labeled_at),
        "featuresSnapshot": event.features_snapshot or {},
        "resultSnapshot": event.result_snapshot or {},
        "createdAt": _iso(event.created_at),
        "updatedAt": _iso(event.updated_at),
    }


def _request_out(
    request: QuickMatchRequest,
    *,
    service_name: str | None = None,
    include_candidates: bool = False,
) -> dict[str, Any]:
    user = getattr(request, "user", None)
    matched_party = getattr(request, "matched_party", None)
    candidates = getattr(request, "candidates", None) or []

    return {
        "requestId": str(request.id),
        "userId": str(request.user_id),
        "userNickname": getattr(user, "nickname", None) or str(request.user_id),
        "serviceId": str(request.service_id),
        "serviceName": service_name or str(request.service_id),
        "status": _value(request.status),
        "retryCount": int(request.retry_count or 0),
        "preferredConditions": request.preferred_conditions or {},
        "matchedPartyId": str(request.matched_party_id) if request.matched_party_id else None,
        "matchedPartyName": _party_name(matched_party, request.matched_party_id),
        "failReason": request.fail_reason,
        "requestProfileSnapshot": request.request_profile_snapshot or {},
        "requestedAt": _iso(request.requested_at),
        "matchedAt": _iso(request.matched_at),
        "expiredAt": _iso(request.expired_at),
        "createdAt": _iso(request.created_at),
        "updatedAt": _iso(request.updated_at),
        "isActive": bool(request.is_active),
        "totalMatchSeconds": _elapsed_seconds(request.requested_at, request.matched_at or request.updated_at),
        "candidateCount": len(candidates),
        "selectedCandidate": next(
            (_candidate_out(candidate) for candidate in candidates if _value(candidate.status) == "selected"),
            None,
        ),
        "candidates": [_candidate_out(candidate) for candidate in candidates] if include_candidates else [],
    }


@router.get("/summary")
async def get_admin_quick_match_summary(
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    today_start = datetime.combine(date.today(), time.min)
    today_end = datetime.combine(date.today(), time.max)

    total = int(await db.scalar(select(func.count(QuickMatchRequest.id))) or 0)
    today_total = int(
        await db.scalar(
            select(func.count(QuickMatchRequest.id)).where(
                QuickMatchRequest.requested_at >= today_start,
                QuickMatchRequest.requested_at <= today_end,
            )
        )
        or 0
    )

    status_rows = (
        await db.execute(
            select(QuickMatchRequest.status, func.count(QuickMatchRequest.id)).group_by(QuickMatchRequest.status)
        )
    ).all()
    status_counts = {_value(status): int(count or 0) for status, count in status_rows}

    label_rows = (
        await db.execute(
            select(QuickMatchTrainingEvent.label_status, func.count(QuickMatchTrainingEvent.id)).group_by(
                QuickMatchTrainingEvent.label_status
            )
        )
    ).all()
    label_counts = {_value(status): int(count or 0) for status, count in label_rows}

    global_stat = (
        await db.execute(
            select(QuickMatchTrainingStat).where(
                QuickMatchTrainingStat.stat_type == "global",
                QuickMatchTrainingStat.stat_key == "all",
            )
        )
    ).scalar_one_or_none()

    matched = status_counts.get("matched", 0)
    failed = status_counts.get("failed", 0)

    return {
        "requests": {
            "total": total,
            "todayTotal": today_total,
            "matched": matched,
            "failed": failed,
            "expired": status_counts.get("expired", 0),
            "requested": status_counts.get("requested", 0),
            "matchRate": round((matched / total) * 100, 2) if total else 0.0,
        },
        "training": {
            "labelCounts": label_counts,
            "successRate": float(global_stat.success_rate or 0) if global_stat else 0.0,
            "sampleCount": int(global_stat.total_count or 0) if global_stat else 0,
            "lastGeneratedAt": _iso(global_stat.generated_at) if global_stat else None,
        },
    }


@router.get("/requests")
async def get_admin_quick_match_requests(
    keyword: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    date_from: date | None = Query(default=None, alias="dateFrom"),
    date_to: date | None = Query(default=None, alias="dateTo"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100, alias="pageSize"),
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    start_date, end_date = _date_bounds(date_from, date_to)
    stmt = (
        select(QuickMatchRequest)
        .options(
            selectinload(QuickMatchRequest.user),
            selectinload(QuickMatchRequest.matched_party),
            selectinload(QuickMatchRequest.candidates).selectinload(QuickMatchCandidate.party),
        )
        .order_by(QuickMatchRequest.requested_at.desc())
    )

    if status_filter and status_filter != "all":
        stmt = stmt.where(cast(QuickMatchRequest.status, String) == status_filter)
    if start_date:
        stmt = stmt.where(QuickMatchRequest.requested_at >= start_date)
    if end_date:
        stmt = stmt.where(QuickMatchRequest.requested_at <= end_date)
    if keyword:
        like = f"%{keyword}%"
        stmt = stmt.join(User, User.id == QuickMatchRequest.user_id)
        stmt = stmt.where(
            or_(
                cast(QuickMatchRequest.id, String).ilike(like),
                cast(QuickMatchRequest.user_id, String).ilike(like),
                User.nickname.ilike(like),
            )
        )

    all_rows = (await db.execute(stmt)).scalars().unique().all()
    total = len(all_rows)
    page_rows = all_rows[(page - 1) * page_size : page * page_size]
    service_map = await _service_name_map(db, {row.service_id for row in page_rows})

    return {
        "rows": [
            _request_out(row, service_name=service_map.get(str(row.service_id)))
            for row in page_rows
        ],
        "total": total,
        "page": page,
        "pageSize": page_size,
    }


@router.get("/requests/{request_id}")
async def get_admin_quick_match_request_detail(
    request_id: uuid.UUID,
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    request = (
        await db.execute(
            select(QuickMatchRequest)
            .options(
                selectinload(QuickMatchRequest.user),
                selectinload(QuickMatchRequest.matched_party),
                selectinload(QuickMatchRequest.candidates).selectinload(QuickMatchCandidate.party),
            )
            .where(QuickMatchRequest.id == request_id)
        )
    ).scalar_one_or_none()
    if not request:
        raise HTTPException(status_code=404, detail="빠른매칭 요청을 찾을 수 없습니다.")

    service_map = await _service_name_map(db, {request.service_id})
    result = (
        await db.execute(select(QuickMatchResult).where(QuickMatchResult.request_id == request_id))
    ).scalar_one_or_none()
    events = (
        await db.execute(
            select(QuickMatchTrainingEvent)
            .where(QuickMatchTrainingEvent.request_id == request_id)
            .order_by(QuickMatchTrainingEvent.created_at.asc())
        )
    ).scalars().all()

    return {
        "request": _request_out(
            request,
            service_name=service_map.get(str(request.service_id)),
            include_candidates=True,
        ),
        "result": {
            "resultId": str(result.id),
            "selectedPartyId": str(result.selected_party_id) if result.selected_party_id else None,
            "selectedCandidateId": str(result.selected_candidate_id) if result.selected_candidate_id else None,
            "decisionReason": result.decision_reason,
            "requestSnapshot": result.request_snapshot or {},
            "candidateSnapshot": result.candidate_snapshot or {},
            "finalScores": result.final_scores or {},
            "createdAt": _iso(result.created_at),
        }
        if result
        else None,
        "trainingEvents": [_training_event_out(event) for event in events],
    }


@router.get("/training-events")
async def get_admin_quick_match_training_events(
    label_status: str | None = Query(default=None, alias="labelStatus"),
    label_reason: str | None = Query(default=None, alias="labelReason"),
    seed_only: bool = Query(default=False, alias="seedOnly"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100, alias="pageSize"),
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(QuickMatchTrainingEvent).order_by(QuickMatchTrainingEvent.created_at.desc())
    if label_status and label_status != "all":
        stmt = stmt.where(QuickMatchTrainingEvent.label_status == label_status)
    if label_reason:
        stmt = stmt.where(QuickMatchTrainingEvent.label_reason == label_reason)
    if seed_only:
        stmt = stmt.where(
            QuickMatchTrainingEvent.features_snapshot["seed"]["source"].as_string().isnot(None)
        )

    all_rows = (await db.execute(stmt)).scalars().all()
    total = len(all_rows)
    page_rows = all_rows[(page - 1) * page_size : page * page_size]

    return {
        "rows": [_training_event_out(event) for event in page_rows],
        "total": total,
        "page": page,
        "pageSize": page_size,
    }


@router.get("/training-stats")
async def get_admin_quick_match_training_stats(
    stat_type: str | None = Query(default=None, alias="statType"),
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(QuickMatchTrainingStat).order_by(
        QuickMatchTrainingStat.stat_type.asc(),
        QuickMatchTrainingStat.stat_key.asc(),
    )
    if stat_type and stat_type != "all":
        stmt = stmt.where(QuickMatchTrainingStat.stat_type == stat_type)

    rows = (await db.execute(stmt)).scalars().all()
    summary = {
        "statCount": len(rows),
        "globalSuccessRate": 0.0,
        "globalSampleCount": 0,
        "lastGeneratedAt": None,
    }

    items = []
    for row in rows:
        generated_at = _iso(row.generated_at)
        if row.stat_type == "global" and row.stat_key == "all":
            summary["globalSuccessRate"] = float(row.success_rate or 0)
            summary["globalSampleCount"] = int(row.total_count or 0)
            summary["lastGeneratedAt"] = generated_at

        items.append(
            {
                "statId": str(row.id),
                "statType": row.stat_type,
                "statKey": row.stat_key,
                "successCount": int(row.success_count or 0),
                "failedCount": int(row.failed_count or 0),
                "totalCount": int(row.total_count or 0),
                "successRate": float(row.success_rate or 0),
                "generatedAt": generated_at,
                "metadata": row.metadata_snapshot or {},
            }
        )

    return {"summary": summary, "rows": items}


@router.get("/quality")
async def get_admin_quick_match_quality(
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    label_rows = (
        await db.execute(
            select(
                QuickMatchTrainingEvent.label_status,
                QuickMatchTrainingEvent.label_reason,
                func.count(QuickMatchTrainingEvent.id),
            ).group_by(QuickMatchTrainingEvent.label_status, QuickMatchTrainingEvent.label_reason)
        )
    ).all()

    failed_total = 0
    success_total = 0
    excluded_total = 0
    pending_total = 0
    reasons = []

    for status, reason, count in label_rows:
        status_value = _value(status)
        count_value = int(count or 0)
        if status_value == "success":
            success_total += count_value
        elif status_value == "failed":
            failed_total += count_value
        elif status_value == "excluded":
            excluded_total += count_value
        elif status_value == "pending":
            pending_total += count_value

        reasons.append(
            {
                "labelStatus": status_value,
                "labelReason": reason,
                "count": count_value,
            }
        )

    trainable_total = success_total + failed_total
    return {
        "summary": {
            "success": success_total,
            "failed": failed_total,
            "pending": pending_total,
            "excluded": excluded_total,
            "trainableTotal": trainable_total,
            "successRate": round((success_total / trainable_total) * 100, 2)
            if trainable_total
            else 0.0,
        },
        "reasonDistribution": sorted(
            reasons,
            key=lambda item: (item["labelStatus"], item["labelReason"] or "", -item["count"]),
        ),
    }


@router.post("/training-stats/rebuild")
async def rebuild_admin_quick_match_training_stats(
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    result = await rebuild_training_stats(db)
    await db.commit()
    return {"success": True, "result": result}


@router.post("/training-label/run")
async def run_admin_quick_match_training_label(
    retention_days: int = Query(default=30, ge=1, le=365, alias="retentionDays"),
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    label_result = await label_retained_successes(db, retention_days=retention_days)
    stats_result = await rebuild_training_stats(db)
    await db.commit()
    return {
        "success": True,
        "labelResult": label_result,
        "statsResult": stats_result,
    }


@router.post("/requests/{request_id}/retry")
async def retry_admin_quick_match_request(
    request_id: uuid.UUID,
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    result = await quick_match_service.retry_match(db=db, request_id=request_id)
    return {"success": True, "result": result}


@router.post("/requests/{request_id}/force-fail")
async def force_fail_admin_quick_match_request(
    request_id: uuid.UUID,
    reason: str = Query(default="ADMIN_FORCE_FAILED"),
    _: AdminContext = Depends(require_admin_quick_match_permission),
    db: AsyncSession = Depends(get_db),
):
    result = await quick_match_service.fail_request(db=db, request_id=request_id, reason=reason)
    return {"success": True, "requestId": str(result.id), "status": _value(result.status)}
