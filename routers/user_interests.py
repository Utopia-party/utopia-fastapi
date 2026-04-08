import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import require_user
from models.user import User
from models.user_interest import UserInterest
from models.party import Service
from schemas.user_interest import UserInterestListResponse, UserInterestUpdateRequest

router = APIRouter(tags=["user-interests"])


@router.get("/users/me/interests", response_model=UserInterestListResponse)
async def get_my_interests(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserInterest)
        .where(UserInterest.user_id == current_user.id)
        .order_by(UserInterest.created_at.asc())
    )
    items = [str(interest.service_id) for interest in result.scalars().all()]
    return UserInterestListResponse(items=items)


@router.put("/users/me/interests", response_model=UserInterestListResponse)
async def update_my_interests(
    payload: UserInterestUpdateRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    # service_id 유효성 검사
    service_ids: list[uuid.UUID] = []
    for raw_id in payload.items:
        try:
            service_ids.append(uuid.UUID(str(raw_id)))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 service_id: {raw_id}")

    # 중복 제거 (순서 유지)
    seen: set[uuid.UUID] = set()
    unique_ids: list[uuid.UUID] = []
    for sid in service_ids:
        if sid not in seen:
            seen.add(sid)
            unique_ids.append(sid)

    # 기존 관심사 전체 삭제 후 재삽입
    await db.execute(delete(UserInterest).where(UserInterest.user_id == current_user.id))
    for sid in unique_ids:
        db.add(UserInterest(user_id=current_user.id, service_id=sid))

    await db.commit()
    return UserInterestListResponse(items=[str(sid) for sid in unique_ids])

