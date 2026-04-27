import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import get_db
from models.party import Party, PartyMember
from models.user import User
from models.user_praise import UserPraise


router = APIRouter(prefix="/praises", tags=["praises"])

PRAISE_COOLDOWN_DAYS = 30

PraiseType = Literal[
    "kind",
    "fast_response",
    "responsible",
    "good_mood",
    "custom",
]


class PraiseCreateRequest(BaseModel):
    to_user_id: uuid.UUID
    praise_type: PraiseType
    message: str | None = Field(default=None, max_length=120)

    # user_praises 테이블에는 저장하지 않고,
    # 채팅방에서 칭찬하는 상황인지 검증하기 위한 값
    party_id: uuid.UUID | None = None


class PraiseResponse(BaseModel):
    id: uuid.UUID
    from_user_id: uuid.UUID
    to_user_id: uuid.UUID
    praise_type: str
    message: str | None
    created_at: datetime


class PraiseAvailabilityResponse(BaseModel):
    can_praise: bool
    last_praised_at: datetime | None = None
    next_available_at: datetime | None = None
    remaining_days: int = 0


def _now_naive_utc() -> datetime:
    """
    user_praises.created_at이 timestamp without time zone 이므로
    비교용 시간도 naive UTC로 맞춘다.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _next_available_at(created_at: datetime) -> datetime:
    return created_at + timedelta(days=PRAISE_COOLDOWN_DAYS)


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    기존 websocket_chat 코드가 access_token 쿠키를 직접 읽는 방식이라,
    HTTP 라우터도 같은 방식으로 맞춘 버전입니다.

    프로젝트에 이미 get_current_user 의존성이 있다면 이 함수는 제거하고
    기존 의존성으로 교체해도 됩니다.
    """
    access_token = request.cookies.get("access_token")

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다.",
        )

    try:
        payload = jwt.decode(
            access_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )

        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 토큰입니다.",
            )

        user_id = uuid.UUID(payload.get("sub", ""))

    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자를 찾을 수 없습니다.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="비활성화된 계정입니다.",
        )

    return user


async def _get_user_or_404(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="칭찬할 사용자를 찾을 수 없습니다.",
        )

    return user


async def _is_active_party_participant(
    db: AsyncSession,
    party: Party,
    user_id: uuid.UUID,
) -> bool:
    if party.leader_id == user_id:
        return True

    result = await db.execute(
        select(PartyMember).where(
            PartyMember.party_id == party.id,
            PartyMember.user_id == user_id,
            PartyMember.status == "active",
        )
    )

    return result.scalar_one_or_none() is not None


async def _assert_same_party_context(
    db: AsyncSession,
    party_id: uuid.UUID | None,
    from_user_id: uuid.UUID,
    to_user_id: uuid.UUID,
) -> None:
    """
    user_praises 테이블에는 party_id가 없으므로,
    party_id는 '이 채팅방에서 칭찬 가능한 관계인지' 검증만 한다.
    """
    if party_id is None:
        return

    result = await db.execute(select(Party).where(Party.id == party_id))
    party = result.scalar_one_or_none()

    if not party:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="파티를 찾을 수 없습니다.",
        )

    from_user_in_party = await _is_active_party_participant(
        db=db,
        party=party,
        user_id=from_user_id,
    )

    to_user_in_party = await _is_active_party_participant(
        db=db,
        party=party,
        user_id=to_user_id,
    )

    if not from_user_in_party or not to_user_in_party:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="같은 파티의 활성 멤버에게만 칭찬할 수 있습니다.",
        )


async def _get_last_praise(
    db: AsyncSession,
    from_user_id: uuid.UUID,
    to_user_id: uuid.UUID,
) -> UserPraise | None:
    result = await db.execute(
        select(UserPraise)
        .where(
            UserPraise.from_user_id == from_user_id,
            UserPraise.to_user_id == to_user_id,
        )
        .order_by(UserPraise.created_at.desc())
        .limit(1)
    )

    return result.scalar_one_or_none()


def _build_availability_response(
    last_praise: UserPraise | None,
) -> PraiseAvailabilityResponse:
    if not last_praise:
        return PraiseAvailabilityResponse(can_praise=True)

    now = _now_naive_utc()
    next_at = _next_available_at(last_praise.created_at)

    if now >= next_at:
        return PraiseAvailabilityResponse(
            can_praise=True,
            last_praised_at=last_praise.created_at,
            next_available_at=next_at,
            remaining_days=0,
        )

    remaining_seconds = (next_at - now).total_seconds()
    remaining_days = max(1, math.ceil(remaining_seconds / 86400))

    return PraiseAvailabilityResponse(
        can_praise=False,
        last_praised_at=last_praise.created_at,
        next_available_at=next_at,
        remaining_days=remaining_days,
    )


@router.get(
    "/availability/{to_user_id}",
    response_model=PraiseAvailabilityResponse,
)
async def get_praise_availability(
    to_user_id: uuid.UUID,
    party_id: uuid.UUID | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.id == to_user_id:
        return PraiseAvailabilityResponse(can_praise=False)

    await _get_user_or_404(db, to_user_id)

    await _assert_same_party_context(
        db=db,
        party_id=party_id,
        from_user_id=current_user.id,
        to_user_id=to_user_id,
    )

    last_praise = await _get_last_praise(
        db=db,
        from_user_id=current_user.id,
        to_user_id=to_user_id,
    )

    return _build_availability_response(last_praise)


@router.post(
    "",
    response_model=PraiseResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_praise(
    payload: PraiseCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.id == payload.to_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="자기 자신은 칭찬할 수 없습니다.",
        )

    await _get_user_or_404(db, payload.to_user_id)

    await _assert_same_party_context(
        db=db,
        party_id=payload.party_id,
        from_user_id=current_user.id,
        to_user_id=payload.to_user_id,
    )

    last_praise = await _get_last_praise(
        db=db,
        from_user_id=current_user.id,
        to_user_id=payload.to_user_id,
    )

    availability = _build_availability_response(last_praise)

    if not availability.can_praise:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "이미 최근 30일 안에 칭찬한 사용자입니다.",
                "last_praised_at": availability.last_praised_at.isoformat()
                if availability.last_praised_at
                else None,
                "next_available_at": availability.next_available_at.isoformat()
                if availability.next_available_at
                else None,
                "remaining_days": availability.remaining_days,
            },
        )

    praise = UserPraise(
        from_user_id=current_user.id,
        to_user_id=payload.to_user_id,
        praise_type=payload.praise_type,
        message=payload.message.strip() if payload.message else None,
    )

    db.add(praise)

    try:
        await db.commit()
        await db.refresh(praise)

    except IntegrityError:
        await db.rollback()

        # 동시 요청 등으로 DB exclusion constraint에 걸린 경우
        latest = await _get_last_praise(
            db=db,
            from_user_id=current_user.id,
            to_user_id=payload.to_user_id,
        )
        latest_availability = _build_availability_response(latest)

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "이미 최근 30일 안에 칭찬한 사용자입니다.",
                "last_praised_at": latest_availability.last_praised_at.isoformat()
                if latest_availability.last_praised_at
                else None,
                "next_available_at": latest_availability.next_available_at.isoformat()
                if latest_availability.next_available_at
                else None,
                "remaining_days": latest_availability.remaining_days,
            },
        )

    return PraiseResponse(
        id=praise.id,
        from_user_id=praise.from_user_id,
        to_user_id=praise.to_user_id,
        praise_type=praise.praise_type,
        message=praise.message,
        created_at=praise.created_at,
    )