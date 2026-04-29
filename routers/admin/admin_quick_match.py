from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import String, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database import get_db
from core.security import require_user
from models.admin import AdminRole
from models.party import Party
from models.quick_match.candidate import QuickMatchCandidate
from models.quick_match.embedding import PartyMatchEmbedding
from models.quick_match.request import QuickMatchRequest
from models.user import User
from schemas.admin_quick_match import (
    AdminQuickMatchActionResponse,
    AdminQuickMatchCandidateOut,
    AdminQuickMatchListOut,
    AdminQuickMatchPolicyOut,
    AdminQuickMatchPolicyResponse,
    AdminQuickMatchProfileSnapshotOut,
    AdminQuickMatchRequestOut,
    AdminQuickMatchSummaryOut,
    StepTimingsOut,
)
from services.quick_match.embedding_service import EmbeddingService
from services.quick_match.party_embedding_service import PartyEmbeddingService
from services.quick_match.quick_match_service import QuickMatchService

router = APIRouter(
    prefix="/admin/quick-match",
    tags=["Admin Quick Match"],
)

quick_match_service = QuickMatchService()
party_embedding_service = PartyEmbeddingService()

CURRENT_POLICY = AdminQuickMatchPolicyOut()


async def require_admin_user(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_user),
) -> User:
    result = await db.execute(
        select(AdminRole).where(AdminRole.user_id == current_user.id)
    )
    role = result.scalar_one_or_none()

    if not role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 필요합니다.",
        )

    if not (
        role.can_manage_quick_match
        or role.can_manage_admins
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="빠른매칭 관리 권한이 없습니다.",
        )

    return current_user


def enum_upper(value: object) -> str:
    raw = value.value if hasattr(value, "value") else str(value)
    return raw.upper()


def candidate_status_upper(value: object) -> str:
    raw = enum_upper(value)
    if raw == "SKIPPED":
        return "FAILED"
    return raw


def to_iso_text(value: datetime | None) -> str:
    if not value:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def get_total_match_seconds(row: QuickMatchRequest) -> float | None:
    if not row.requested_at:
        return None

    end_at = row.matched_at or row.updated_at
    if not end_at:
        return None

    start = row.requested_at
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=timezone.utc)

    return round(max((end_at - start).total_seconds(), 0), 2)


def build_step_timings(row: QuickMatchRequest) -> StepTimingsOut:
    return StepTimingsOut()


def normalize_profile_snapshot(
    snapshot: dict | None,
) -> AdminQuickMatchProfileSnapshotOut:
    data = snapshot or {}
    preferred = data.get("preferred_conditions") or {}
    activity = data.get("activity_summary") or {}
    payment = data.get("payment_summary") or {}
    risk = data.get("risk_summary") or {}

    return AdminQuickMatchProfileSnapshotOut(
        trustScore=float(data.get("trust_score") or 0),
        preferredConditions={
            "category": preferred.get("category"),
            "platform": preferred.get("platform"),
            "durationPreference": preferred.get("duration_preference"),
        },
        activitySummary={
            "totalPartyJoinCount": int(activity.get("total_party_join_count") or 0),
            "servicePartyJoinCount": int(activity.get("service_party_join_count") or 0),
            "activePartyCount": int(activity.get("active_party_count") or 0),
        },
        paymentSummary={
            "settlementSuccessCount": int(payment.get("settlement_success_count") or 0),
        },
        riskSummary={
            "reportCount": int(risk.get("report_count") or 0),
            "leaveCount": int(risk.get("leave_count") or 0),
            "isCurrentlyBanned": bool(risk.get("is_currently_banned") or False),
        },
    )


def build_candidate_out(candidate: QuickMatchCandidate) -> AdminQuickMatchCandidateOut:
    party = getattr(candidate, "party", None)
    party_name = (
        getattr(party, "title", None)
        or getattr(party, "name", None)
        or getattr(party, "party_name", None)
        or str(candidate.party_id)
    )

    return AdminQuickMatchCandidateOut(
        candidateId=str(candidate.id),
        partyId=str(candidate.party_id),
        partyName=party_name,
        rank=candidate.rank,
        status=candidate_status_upper(candidate.status),
        ruleScore=float(candidate.rule_score or 0),
        vectorScore=float(candidate.vector_score or 0),
        finalScore=float(candidate.ai_score or 0),
        filterReasons=candidate.filter_reasons or {},
    )


def build_request_out(row: QuickMatchRequest) -> AdminQuickMatchRequestOut:
    user = getattr(row, "user", None)
    matched_party = getattr(row, "matched_party", None)
    service = getattr(matched_party, "service", None)

    service_name = (
        getattr(service, "name", None)
        or getattr(row, "service_name", None)
        or str(row.service_id)
    )

    matched_party_name = None
    if matched_party:
        matched_party_name = (
            getattr(matched_party, "title", None)
            or getattr(matched_party, "name", None)
            or getattr(matched_party, "party_name", None)
            or str(matched_party.id)
        )

    return AdminQuickMatchRequestOut(
        requestId=str(row.id),
        requestedAt=to_iso_text(row.requested_at),
        userId=str(row.user_id),
        userNickname=(
            getattr(user, "nickname", None)
            or getattr(user, "name", None)
            or str(row.user_id)
        ),
        serviceName=service_name,
        status=enum_upper(row.status),
        matchedPartyId=str(row.matched_party_id) if row.matched_party_id else None,
        matchedPartyName=matched_party_name,
        totalMatchSeconds=get_total_match_seconds(row),
        retryCount=int(row.retry_count or 0),
        failReason=row.fail_reason,
        stepTimings=build_step_timings(row),
        aiProfileSnapshot=normalize_profile_snapshot(row.ai_profile_snapshot),
        candidates=[build_candidate_out(c) for c in (row.candidates or [])],
    )


async def get_service_name_map(
    db: AsyncSession,
    service_ids: set[uuid.UUID],
) -> dict[str, str]:
    if not service_ids:
        return {}

    from models.party import Service

    result = await db.execute(
        select(Service.id, Service.name).where(Service.id.in_(service_ids))
    )

    return {str(service_id): name for service_id, name in result.all()}


async def build_summary(db: AsyncSession) -> AdminQuickMatchSummaryOut:
    total_result = await db.execute(select(func.count(QuickMatchRequest.id)))
    total = int(total_result.scalar() or 0)

    today_result = await db.execute(
        select(func.count(QuickMatchRequest.id)).where(
            func.date(QuickMatchRequest.requested_at) == date.today()
        )
    )
    today_total = int(today_result.scalar() or 0)

    matched_result = await db.execute(
        select(func.count(QuickMatchRequest.id)).where(
            func.upper(cast(QuickMatchRequest.status, String)).in_(
                ["MATCHED", "REMATCHING"]
            )
        )
    )
    matched = int(matched_result.scalar() or 0)

    rows_result = await db.execute(
        select(QuickMatchRequest).where(
            QuickMatchRequest.matched_at.is_not(None)
        )
    )
    rows = rows_result.scalars().all()

    elapsed_values = [
        value
        for value in (get_total_match_seconds(row) for row in rows)
        if value is not None
    ]
    avg_seconds = (
        round(sum(elapsed_values) / len(elapsed_values), 2)
        if elapsed_values
        else 0
    )

    success_rate = round((matched / total) * 100, 1) if total else 0

    return AdminQuickMatchSummaryOut(
        total=total,
        todayTotal=today_total,
        matched=matched,
        successRate=success_rate,
        avgSeconds=avg_seconds,
        stepAvg=StepTimingsOut(),
    )


@router.get("/requests", response_model=AdminQuickMatchListOut)
async def get_admin_quick_match_requests(
    keyword: str | None = Query(default=None),
    status_filter: str | None = Query(default=None, alias="status"),
    service_name: str | None = Query(default=None, alias="serviceName"),
    date_from: str | None = Query(default=None, alias="dateFrom"),
    date_to: str | None = Query(default=None, alias="dateTo"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100, alias="pageSize"),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin_user),
):
    stmt = (
        select(QuickMatchRequest)
        .options(
            selectinload(QuickMatchRequest.user),
            selectinload(QuickMatchRequest.matched_party).selectinload(Party.service),
            selectinload(QuickMatchRequest.candidates).selectinload(
                QuickMatchCandidate.party
            ),
        )
        .order_by(QuickMatchRequest.requested_at.desc())
    )

    if status_filter and status_filter != "전체":
        stmt = stmt.where(
            func.upper(cast(QuickMatchRequest.status, String)) == status_filter.upper()
        )

    if date_from:
        stmt = stmt.where(
            QuickMatchRequest.requested_at >= datetime.fromisoformat(date_from)
        )

    if date_to:
        end = datetime.fromisoformat(date_to).replace(
            hour=23,
            minute=59,
            second=59,
        )
        stmt = stmt.where(QuickMatchRequest.requested_at <= end)

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

    all_result = await db.execute(stmt)
    all_rows = all_result.scalars().unique().all()

    if service_name and service_name != "전체":
        service_map = await get_service_name_map(
            db,
            {row.service_id for row in all_rows},
        )
        all_rows = [
            row
            for row in all_rows
            if service_map.get(str(row.service_id)) == service_name
        ]

    total = len(all_rows)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = all_rows[start:end]

    service_map = await get_service_name_map(
        db,
        {row.service_id for row in page_rows},
    )

    response_rows = []
    for row in page_rows:
        out = build_request_out(row)
        out.serviceName = service_map.get(str(row.service_id), out.serviceName)
        response_rows.append(out)

    return AdminQuickMatchListOut(
        summary=await build_summary(db),
        rows=response_rows,
        total=total,
        page=page,
        pageSize=page_size,
    )


@router.get("/requests/{request_id}", response_model=AdminQuickMatchRequestOut)
async def get_admin_quick_match_request_detail(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin_user),
):
    result = await db.execute(
        select(QuickMatchRequest)
        .options(
            selectinload(QuickMatchRequest.user),
            selectinload(QuickMatchRequest.matched_party).selectinload(Party.service),
            selectinload(QuickMatchRequest.candidates).selectinload(
                QuickMatchCandidate.party
            ),
        )
        .where(QuickMatchRequest.id == request_id)
    )
    row = result.scalar_one_or_none()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="빠른매칭 요청을 찾을 수 없습니다.",
        )

    service_map = await get_service_name_map(db, {row.service_id})
    out = build_request_out(row)
    out.serviceName = service_map.get(str(row.service_id), out.serviceName)
    return out


@router.get("/policy", response_model=AdminQuickMatchPolicyResponse)
async def get_admin_quick_match_policy(
    _: User = Depends(require_admin_user),
):
    return AdminQuickMatchPolicyResponse(policy=CURRENT_POLICY)


@router.patch("/policy", response_model=AdminQuickMatchPolicyResponse)
async def update_admin_quick_match_policy(
    payload: AdminQuickMatchPolicyOut,
    _: User = Depends(require_admin_user),
):
    global CURRENT_POLICY
    CURRENT_POLICY = payload
    return AdminQuickMatchPolicyResponse(policy=CURRENT_POLICY)


@router.post(
    "/requests/{request_id}/retry",
    response_model=AdminQuickMatchActionResponse,
)
async def retry_admin_quick_match_request(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin_user),
):
    await quick_match_service.retry_match(db=db, request_id=request_id)
    return AdminQuickMatchActionResponse(
        success=True,
        message="재시도 요청이 처리되었습니다.",
    )


@router.post(
    "/requests/{request_id}/force-fail",
    response_model=AdminQuickMatchActionResponse,
)
async def force_fail_admin_quick_match_request(
    request_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin_user),
):
    await quick_match_service.fail_request(
        db=db,
        request_id=request_id,
        reason="ADMIN_FORCE_FAILED",
    )
    return AdminQuickMatchActionResponse(
        success=True,
        message="요청을 강제 실패 처리했습니다.",
    )


@router.post(
    "/users/{user_id}/embedding/regenerate",
    response_model=AdminQuickMatchActionResponse,
)
async def regenerate_user_quick_match_embedding(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin_user),
):
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    requests_result = await db.execute(
        select(QuickMatchRequest)
        .where(QuickMatchRequest.user_id == user_id)
        .order_by(QuickMatchRequest.requested_at.desc())
        .limit(1)
    )
    latest_request = requests_result.scalar_one_or_none()

    if not latest_request:
        raise HTTPException(status_code=404, detail="빠른매칭 요청 이력이 없습니다.")

    ai_profile = await quick_match_service._build_user_ai_profile(
        db=db,
        user=user,
        service_id=latest_request.service_id,
        preferred_conditions=latest_request.preferred_conditions or {},
    )

    embedding_text = EmbeddingService.serialize_user_profile_text(ai_profile)
    embedding_vector = await EmbeddingService.generate_embedding(
        {"text": embedding_text}
    )

    result = await db.execute(
        select(PartyMatchEmbedding).where(
            PartyMatchEmbedding.user_id == user_id,
            PartyMatchEmbedding.service_id == latest_request.service_id,
        )
    )
    embedding = result.scalar_one_or_none()

    if embedding:
        embedding.embedding_vector = embedding_vector
        embedding.source_snapshot = ai_profile
        embedding.last_generated_at = datetime.now(timezone.utc)
    else:
        embedding = PartyMatchEmbedding(
            user_id=user_id,
            service_id=latest_request.service_id,
            embedding_vector=embedding_vector,
            source_snapshot=ai_profile,
            last_generated_at=datetime.now(timezone.utc),
        )
        db.add(embedding)

    await db.commit()

    return AdminQuickMatchActionResponse(
        success=True,
        message="사용자 임베딩을 재생성했습니다.",
    )


@router.post(
    "/parties/{party_id}/embedding/regenerate",
    response_model=AdminQuickMatchActionResponse,
)
async def regenerate_party_quick_match_embedding(
    party_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin_user),
):
    embedding = await party_embedding_service.sync_party_embedding(
        db=db,
        party_id=party_id,
    )

    if not embedding:
        raise HTTPException(
            status_code=404,
            detail="파티 임베딩을 생성하지 못했습니다.",
        )

    await db.commit()

    return AdminQuickMatchActionResponse(
        success=True,
        message="파티 임베딩을 재생성했습니다.",
    )


@router.post(
    "/embedding-backfill",
    response_model=AdminQuickMatchActionResponse,
)
async def run_quick_match_embedding_backfill(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_admin_user),
):
    result = await db.execute(
        select(Party)
        .options(selectinload(Party.service))
        .where(Party.status == "recruiting")
    )
    parties = result.scalars().all()

    count = 0
    for party in parties:
        embedding = await party_embedding_service.sync_party_embedding(
            db=db,
            party_id=party.id,
        )
        if embedding:
            count += 1

    await db.commit()

    return AdminQuickMatchActionResponse(
        success=True,
        message=f"파티 임베딩 백필 완료: {count}건",
    )
