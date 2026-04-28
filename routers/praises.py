import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from core.database import get_db
from core.security import get_current_user
from models.party import Party, PartyMember
from models.user import User
from models.user_praise import UserPraise
from models.mypage.trust_score import TrustScore
from services.mypage.profile_service import _build_profile_image_url


router = APIRouter(prefix="/praises", tags=["praises"])

PRAISE_COOLDOWN_DAYS = 30
PRAISE_TRUST_DELTA = 0.5
DEFAULT_TRUST_SCORE = 36.5
MAX_TRUST_SCORE = 100.0


PraiseType = Literal[
    "kind",
    "fast_response",
    "responsible",
    "good_mood",
    "custom",
]

PraiseDirection = Literal["received", "sent"]


class PraiseCreateRequest(BaseModel):
    to_user_id: uuid.UUID
    praise_type: PraiseType
    message: str | None = Field(default=None, max_length=120)

    # user_praises 테이블에는 저장하지 않고,
    # 채팅방에서 칭찬 가능한 관계인지 검증하기 위한 값
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


class MyPraiseItemResponse(BaseModel):
    id: uuid.UUID

    from_user_id: uuid.UUID
    from_nickname: str | None = None
    from_profile_image: str | None = None

    to_user_id: uuid.UUID
    to_nickname: str | None = None
    to_profile_image: str | None = None

    praise_type: str
    message: str | None = None
    created_at: datetime


class MyPraisesResponse(BaseModel):
    items: list[MyPraiseItemResponse]
    total: int

class DeletePraiseResponse(BaseModel):
    deleted: bool
    praise_id: uuid.UUID


def _now_naive_utc() -> datetime:
    """
    user_praises.created_at이 timestamp without time zone 이므로
    비교용 시간도 naive UTC로 맞춘다.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _next_available_at(created_at: datetime) -> datetime:
    return created_at + timedelta(days=PRAISE_COOLDOWN_DAYS)


def _safe_profile_image_url(profile_image_key: str | None) -> str | None:
    try:
        return _build_profile_image_url(profile_image_key)
    except Exception:
        return None


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


def _normalize_score(value: object) -> float:
    if value is None:
        return DEFAULT_TRUST_SCORE

    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return DEFAULT_TRUST_SCORE


async def _apply_trust_reward_for_praise(
    db: AsyncSession,
    *,
    to_user: User,
    from_user_id: uuid.UUID,
    praise_id: uuid.UUID,
) -> None:
    previous_score = _normalize_score(to_user.trust_score)

    new_score = round(previous_score + PRAISE_TRUST_DELTA, 1)
    new_score = min(MAX_TRUST_SCORE, new_score)

    change_amount = round(new_score - previous_score, 1)

    # 이미 100점이면 이력에 0.0 상승을 남기지 않음
    if change_amount <= 0:
        return

    to_user.trust_score = new_score

    db.add(
        TrustScore(
            user_id=to_user.id,
            previous_score=previous_score,
            new_score=new_score,
            change_amount=change_amount,
            reason="칭찬을 받아 신뢰도가 상승했습니다.",
            reference_id=praise_id,
            created_by=from_user_id,
        )
    )


@router.get(
    "/me",
    response_model=MyPraisesResponse,
)
async def get_my_praises(
    direction: PraiseDirection = Query(default="received"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    FromUser = aliased(User)
    ToUser = aliased(User)

    if direction == "received":
        condition = (
            (UserPraise.to_user_id == current_user.id)
            & (UserPraise.hidden_from_receiver_at.is_(None))
        )
    else:
        condition = (
            (UserPraise.from_user_id == current_user.id)
            & (UserPraise.hidden_from_sender_at.is_(None))
        )

    total_result = await db.execute(
        select(func.count())
        .select_from(UserPraise)
        .where(condition)
    )
    total = int(total_result.scalar_one() or 0)

    result = await db.execute(
        select(UserPraise, FromUser, ToUser)
        .join(FromUser, UserPraise.from_user_id == FromUser.id)
        .join(ToUser, UserPraise.to_user_id == ToUser.id)
        .where(condition)
        .order_by(UserPraise.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    rows = result.all()

    items = [
        MyPraiseItemResponse(
            id=praise.id,
            from_user_id=praise.from_user_id,
            from_nickname=from_user.nickname,
            from_profile_image=_safe_profile_image_url(
                from_user.profile_image_key,
            ),
            to_user_id=praise.to_user_id,
            to_nickname=to_user.nickname,
            to_profile_image=_safe_profile_image_url(
                to_user.profile_image_key,
            ),
            praise_type=praise.praise_type,
            message=praise.message,
            created_at=praise.created_at,
        )
        for praise, from_user, to_user in rows
    ]

    return MyPraisesResponse(
        items=items,
        total=total,
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

    to_user = await _get_user_or_404(db, payload.to_user_id)

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

    cleaned_message = payload.message.strip() if payload.message else None
    if cleaned_message == "":
        cleaned_message = None

    praise = UserPraise(
        from_user_id=current_user.id,
        to_user_id=payload.to_user_id,
        praise_type=payload.praise_type,
        message=cleaned_message,
    )

    db.add(praise)

    try:
        # commit 전에 user_praises.id를 확보하기 위해 flush
        await db.flush()

        await _apply_trust_reward_for_praise(
            db=db,
            to_user=to_user,
            from_user_id=current_user.id,
            praise_id=praise.id,
        )

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

@router.delete(
    "/{praise_id}",
    response_model=DeletePraiseResponse,
)
async def delete_my_praise(
    praise_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserPraise).where(UserPraise.id == praise_id)
    )
    praise = result.scalar_one_or_none()

    if not praise:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="칭찬 내역을 찾을 수 없습니다.",
        )

    now = _now_naive_utc()

    if praise.from_user_id == current_user.id:
        praise.hidden_from_sender_at = now

    elif praise.to_user_id == current_user.id:
        praise.hidden_from_receiver_at = now

    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="삭제할 권한이 없습니다.",
        )

    await db.commit()

    return DeletePraiseResponse(
        deleted=True,
        praise_id=praise.id,
    )