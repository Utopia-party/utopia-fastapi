from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import require_user
from models.quick_match.candidate import QuickMatchCandidate
from models.quick_match.request import QuickMatchRequest
from models.quick_match.result import QuickMatchResult
from models.user import User
from schemas.quick_match.request import (
    QuickMatchCreateRequest,
    QuickMatchRetryRequest,
)
from schemas.quick_match.response import (
    QuickMatchCreateResponse,
    QuickMatchDetailResponse,
)
from services.quick_match.quick_match_service import QuickMatchService

router = APIRouter(prefix="/quick-match", tags=["quick-match"])

quick_match_service = QuickMatchService()


def _map_quick_match_exception(exc: Exception) -> HTTPException:
    code = str(exc)

    error_map: dict[str, tuple[int, str]] = {
        "USER_NOT_FOUND": (404, "사용자를 찾을 수 없습니다."),
        "USER_INACTIVE": (403, "비활성화된 사용자입니다."),
        "USER_BANNED": (403, "이용이 제한된 사용자입니다."),
        "ALREADY_REQUESTED": (400, "이미 진행 중인 빠른매칭 요청이 있습니다."),
        "ALREADY_IN_ACTIVE_PARTY": (400, "이미 해당 서비스의 활성 파티에 참여 중입니다."),
        "REQUEST_NOT_FOUND": (404, "빠른매칭 요청을 찾을 수 없습니다."),
        "INVALID_REQUEST_STATUS": (400, "현재 상태에서는 후보 탐색을 진행할 수 없습니다."),
        "NO_RECRUITING_PARTY": (404, "현재 모집 중인 파티가 없습니다."),
        "NO_CANDIDATE": (404, "조건에 맞는 후보 파티를 찾지 못했습니다."),
        "PARTY_NOT_FOUND": (404, "파티를 찾을 수 없습니다."),
        "PARTY_STATUS_CHANGED": (409, "파티 상태가 변경되었습니다."),
        "PARTY_FULL": (409, "파티 정원이 가득 찼습니다."),
        "REQUEST_NOT_MATCHED": (400, "매칭 완료 상태의 요청이 아닙니다."),
        "MATCHED_PARTY_NOT_FOUND": (404, "매칭된 파티 정보를 찾을 수 없습니다."),
    }

    status_code, detail = error_map.get(
        code,
        (500, "빠른매칭 처리 중 오류가 발생했습니다."),
    )
    return HTTPException(status_code=status_code, detail=detail)


@router.post(
    "",
    response_model=QuickMatchCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_quick_match(
    body: QuickMatchCreateRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        request = await quick_match_service.create_request(
            db=db,
            user_id=current_user.id,
            service_id=body.service_id,
            preferred_conditions=body.preferred_conditions,
        )

        await quick_match_service.find_candidates(
            db=db,
            request_id=request.id,
        )

        await quick_match_service.select_party(
            db=db,
            request_id=request.id,
        )

        return QuickMatchCreateResponse(
            message="빠른매칭 요청이 생성되었습니다.",
            request_id=request.id,
            status="matched",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _map_quick_match_exception(exc)


@router.get(
    "/{request_id}",
    response_model=QuickMatchDetailResponse,
    status_code=status.HTTP_200_OK,
)
async def get_quick_match_detail(
    request_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    request = await db.get(QuickMatchRequest, request_id)
    if not request:
        raise HTTPException(status_code=404, detail="빠른매칭 요청을 찾을 수 없습니다.")

    if request.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="해당 요청에 접근할 수 없습니다.")

    candidate_result = await db.execute(
        select(QuickMatchCandidate)
        .where(QuickMatchCandidate.request_id == request_id)
        .order_by(QuickMatchCandidate.rank.asc().nullslast(), QuickMatchCandidate.created_at.asc())
    )
    candidates = candidate_result.scalars().all()

    result_result = await db.execute(
        select(QuickMatchResult).where(QuickMatchResult.request_id == request_id)
    )
    result = result_result.scalar_one_or_none()

    return QuickMatchDetailResponse(
        request=request,
        candidates=list(candidates),
        result=result,
    )


@router.post(
    "/{request_id}/join",
    status_code=status.HTTP_200_OK,
)
async def join_quick_matched_party(
    request_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    request = await db.get(QuickMatchRequest, request_id)
    if not request:
        raise HTTPException(status_code=404, detail="빠른매칭 요청을 찾을 수 없습니다.")

    if request.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="해당 요청에 접근할 수 없습니다.")

    try:
        return await quick_match_service.join_party(
            db=db,
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _map_quick_match_exception(exc)


@router.post(
    "/{request_id}/retry",
    status_code=status.HTTP_200_OK,
)
async def retry_quick_match(
    request_id: uuid.UUID,
    body: QuickMatchRetryRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    request = await db.get(QuickMatchRequest, request_id)
    if not request:
        raise HTTPException(status_code=404, detail="빠른매칭 요청을 찾을 수 없습니다.")

    if request.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="해당 요청에 접근할 수 없습니다.")

    try:
        return await quick_match_service.retry_match(
            db=db,
            request_id=request_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _map_quick_match_exception(exc)
