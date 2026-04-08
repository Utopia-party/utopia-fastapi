import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database import get_db
from core.minio_assets import build_minio_asset_url
from core.security import require_user, get_current_user_optional
from models.party import Party, PartyMember, Service
from models.user import User
from schemas.party import (
    CategoryOut,
    PartyCreate,
    PartyListOut,
    PartyOut,
)
# MessageOut 등의 위치가 schemas/party.py가 아니라면 아래 경로를 확인해주세요.
from schemas.user import MessageOut 

router = APIRouter(prefix="/parties", tags=["parties"])
logger = logging.getLogger(__name__)

def _build_party_out(party: Party, current_user_id: Optional[uuid.UUID] = None) -> PartyOut:
    """
    Party 객체를 PartyOut 스키마로 변환하며, 현재 유저의 참여 여부를 계산합니다.
    """
    svc = party.service
    
    # 참여 여부 판별: 방장이거나 멤버 목록에 포함되어 있는지 확인
    is_joined = False
    if current_user_id:
        is_leader = (party.leader_id == current_user_id)
        # members가 로드되어 있는지 확인 후 체크
        is_member = any(m.user_id == current_user_id for m in party.members) if party.members else False
        is_joined = is_leader or is_member

    return PartyOut(
        id=party.id,
        leader_id=party.leader_id,
        service_id=party.service_id,
        title=party.title,
        status=party.status,
        host_nickname=party.host.nickname if party.host else None,
        service_name=svc.name if svc else None,
        category_name=svc.category if svc else None,
        max_members=svc.max_members if svc else None,
        monthly_price=svc.monthly_price if svc else None,
        logo_image_key=svc.logo_image_key if svc else None,
        logo_image_url=build_minio_asset_url(svc.logo_image_key) if svc else None,
        member_count=len(party.members) if party.members is not None else 0,
        is_joined=is_joined
    )

# 422 에러 방지를 위해 추가된 카테고리 목록 엔드포인트
@router.get("/categories", response_model=list[CategoryOut])
async def list_categories(
    db: AsyncSession = Depends(get_db)
):
    """
    등록된 서비스들의 카테고리 목록을 중복 없이 가져옵니다.
    """
    result = await db.execute(select(Service.category).distinct())
    categories = result.scalars().all()
    # 프론트엔드 기대 형식에 맞춰 반환
    return [{"id": uuid.uuid4(), "name": cat} for cat in categories if cat]

@router.get("", response_model=PartyListOut)
async def list_parties(
    category_id: Optional[uuid.UUID] = Query(None),
    service_id: Optional[uuid.UUID] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(12, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    q = (
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.members),
            selectinload(Party.service),
        )
    )

    if service_id:
        q = q.where(Party.service_id == service_id)
    if category_id:
        # 카테고리 ID로 해당 카테고리 이름을 찾아서 필터링
        svc_result = await db.execute(select(Service.category).where(Service.id == category_id))
        cat_name = svc_result.scalar_one_or_none()
        if cat_name:
            q = q.join(Party.service).where(Service.category == cat_name)
    if search:
        q = q.where(Party.title.ilike(f"%{search}%"))

    total = await db.scalar(select(func.count()).select_from(q.subquery())) or 0
    q = q.offset((page - 1) * size).limit(size).order_by(Party.id.desc())
    
    result = await db.execute(q)
    parties = result.scalars().all()

    user_id = current_user.id if current_user else None

    return PartyListOut(
        parties=[_build_party_out(p, user_id) for p in parties],
        total=total,
        page=page,
        size=size
    )

@router.get("/{party_id}", response_model=PartyOut)
async def get_party(
    party_id: uuid.UUID, 
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    result = await db.execute(
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.members),
            selectinload(Party.service),
        )
        .where(Party.id == party_id)
    )
    party = result.scalar_one_or_none()
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")
    
    user_id = current_user.id if current_user else None
    return _build_party_out(party, user_id)

@router.post("", response_model=PartyOut, status_code=status.HTTP_201_CREATED)
async def create_party(
    body: PartyCreate,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = Party(
        leader_id=current_user.id,
        service_id=body.service_id,
        title=body.title,
        status="recruiting",
    )
    db.add(party)
    await db.commit()
    
    result = await db.execute(
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.members),
            selectinload(Party.service)
        )
        .where(Party.id == party.id)
    )
    return _build_party_out(result.scalar_one(), current_user.id)

@router.post("/{party_id}/join", response_model=MessageOut)
async def join_party(
    party_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Party)
        .options(selectinload(Party.service))
        .where(Party.id == party_id)
    )
    party = result.scalar_one_or_none()

    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")
    
    if party.leader_id == current_user.id:
        raise HTTPException(status_code=400, detail="자신이 개설한 파티입니다.")

    existing_check = await db.execute(
        select(PartyMember).where(
            PartyMember.party_id == party_id,
            PartyMember.user_id == current_user.id,
        )
    )
    if existing_check.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 참여한 파티입니다.")

    if party.service:
        count_result = await db.execute(
            select(func.count()).select_from(PartyMember).where(PartyMember.party_id == party_id)
        )
        current_count = count_result.scalar() or 0
        if current_count >= party.service.max_members:
            raise HTTPException(status_code=400, detail="파티 인원이 가득 찼습니다.")

    try:
        new_member = PartyMember(party_id=party_id, user_id=current_user.id)
        db.add(new_member)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error joining party: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="파티 가입 처리 중 서버 오류가 발생했습니다."
        )

    return MessageOut(message="파티 참여가 완료되었습니다.")
