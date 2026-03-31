import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from typing import Optional
from core.database import get_db
from core.security import get_current_user, require_user
from models.party import Party, PartyMember, Service
from models.user import User
from schemas import PartyCreate, PartyOut, PartyListOut, MessageOut, ServiceOut, CategoryOut

router = APIRouter(prefix="/parties", tags=["parties"])


def _build_party_out(party: Party) -> PartyOut:
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
        member_count=len(party.members),
    )


# 프론트 호환용 /categories - 카테고리 중복 제거해서 반환
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


# 서비스 목록 (카테고리 필터 지원)
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
        # category_id는 service.id를 넘기므로 해당 service의 category 문자열로 조인 필터
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

    return PartyListOut(parties=[_build_party_out(p) for p in parties], total=total, page=page, size=size)


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
    party = Party(
        leader_id=current_user.id,
        service_id=body.service_id,
        title=body.title,
        status="RECRUITING",
    )
    db.add(party)
    await db.commit()
    await db.refresh(party)
    result = await db.execute(
        select(Party)
        .options(selectinload(Party.host), selectinload(Party.members), selectinload(Party.service))
        .where(Party.id == party.id)
    )
    return _build_party_out(result.scalar_one())


@router.post("/{party_id}/join", response_model=MessageOut)
async def join_party(
    party_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = await db.get(Party, party_id)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")
    if party.leader_id == current_user.id:
        raise HTTPException(status_code=400, detail="자신이 개설한 파티입니다.")

    existing = await db.execute(
        select(PartyMember).where(
            PartyMember.party_id == party_id,
            PartyMember.user_id == current_user.id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 참여한 파티입니다.")

    # max_members 초과 체크
    if party.service:
        svc_result = await db.execute(
            select(Service).where(Service.id == party.service_id)
        )
        svc = svc_result.scalar_one_or_none()
        if svc:
            count_result = await db.execute(
                select(func.count()).select_from(PartyMember).where(PartyMember.party_id == party_id)
            )
            current_count = count_result.scalar() or 0
            if current_count >= svc.max_members:
                raise HTTPException(status_code=400, detail="파티 인원이 가득 찼습니다.")

    db.add(PartyMember(party_id=party_id, user_id=current_user.id))
    await db.commit()
    return MessageOut(message="파티 참여가 완료되었습니다.")
