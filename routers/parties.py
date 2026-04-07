import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database import get_db
from core.security import require_user
from models.party import Party, PartyMember, Service
from models.user import User
from schemas import (
    CategoryOut,
    MessageOut,
    PartyCreate,
    PartyListOut,
    PartyOut,
    ServiceOut,
)

router = APIRouter(prefix="/parties", tags=["parties"])
logger = logging.getLogger(__name__)

def _build_party_out(party: Party) -> PartyOut:
    """
    Party 객체를 PartyOut 스키마로 변환합니다.
    주의: 호출 전에 service, host, members가 selectinload 되어 있어야 합니다.
    """
    svc = party.service
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
        member_count=len(party.members) if party.members is not None else 0,
    )


@router.get("/categories", response_model=list[CategoryOut], tags=["categories"])
async def list_categories(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Service)
        .where(Service.is_active == True)  # noqa
        .order_by(Service.category, Service.name)
    )
    services = result.scalars().all()
    seen: set[str] = set()
    categories = []
    for s in services:
        if s.category not in seen:
            seen.add(s.category)
            categories.append(CategoryOut(category_id=s.id, category_name=s.category))
    return categories


@router.get("/services", response_model=list[ServiceOut], tags=["services"])
async def list_services(
    category: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    q = select(Service).where(Service.is_active == True).order_by(Service.name)  # noqa
    if category:
        q = q.where(Service.category == category)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("", response_model=PartyListOut)
async def list_parties(
    category_id: Optional[uuid.UUID] = Query(None),
    service_id: Optional[uuid.UUID] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(12, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
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

    return PartyListOut(
        parties=[_build_party_out(p) for p in parties],
        total=total,
        page=page,
        size=size
    )


@router.get("/{party_id}", response_model=PartyOut)
async def get_party(party_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
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
    return _build_party_out(party)


@router.post("", response_model=PartyOut, status_code=status.HTTP_201_CREATED)
async def create_party(
    body: PartyCreate,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    # 파티 생성
    party = Party(
        leader_id=current_user.id,
        service_id=body.service_id,
        title=body.title,
        status="recruiting",
    )
    db.add(party)
    
    # 방장도 멤버로 자동 추가하고 싶다면 여기에 PartyMember 추가 로직을 넣을 수 있습니다.
    
    await db.commit()
    
    # 생성된 파티 정보를 관계 모델과 함께 다시 조회
    result = await db.execute(
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.members),
            selectinload(Party.service)
        )
        .where(Party.id == party.id)
    )
    return _build_party_out(result.scalar_one())


@router.post("/{party_id}/join", response_model=MessageOut)
async def join_party(
    party_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    # 1. 파티 존재 여부와 서비스 정보를 한 번에 조회 (Lazy Loading 방지)
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

    # 2. 이미 참여 중인지 확인
    existing_check = await db.execute(
        select(PartyMember).where(
            PartyMember.party_id == party_id,
            PartyMember.user_id == current_user.id,
        )
    )
    if existing_check.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 참여한 파티입니다.")

    # 3. 정원 초과 여부 확인
    if party.service:
        count_result = await db.execute(
            select(func.count()).select_from(PartyMember).where(PartyMember.party_id == party_id)
        )
        current_count = count_result.scalar() or 0
        if current_count >= party.service.max_members:
            raise HTTPException(status_code=400, detail="파티 인원이 가득 찼습니다.")

    # 4. 멤버 추가
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
