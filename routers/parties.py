import uuid
import logging
from datetime import date
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
    ServiceOut,
)
from schemas.user import MessageOut

router = APIRouter(prefix="/parties", tags=["parties"])
logger = logging.getLogger(__name__)

def _service_monthly_price(service: Service | None) -> int | None:
    if service is None:
        return None
    return service.monthly_price

def _party_max_members(party: Party, service: Service | None) -> int | None:
    return party.max_members or (service.max_members if service else None)

def _party_member_count(party: Party) -> int:
    if party.current_members is not None:
        return party.current_members
    member_count = len(party.members) if party.members is not None else 0
    return member_count + (1 if party.leader_id else 0)

def _service_original_price(service: Service | None) -> int | None:
    if service is None:
        return None
    return service.original_price  

def _build_party_out(party: Party, current_user_id: Optional[uuid.UUID] = None) -> PartyOut:
    svc = party.service
    is_joined = False
    if current_user_id:
        is_leader = (party.leader_id == current_user_id)
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
        max_members=_party_max_members(party, svc),
        monthly_price=party.monthly_per_person,
        original_price=_service_original_price(svc),  
        member_count=_party_member_count(party),
        logo_image_key=svc.logo_image_key if svc else None,
        logo_image_url=build_minio_asset_url(svc.logo_image_key) if svc else None,
        is_joined=is_joined,
    )

@router.get("/services", response_model=list[ServiceOut])
async def list_services(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Service)
        .where(Service.is_active.is_(True))
        .order_by(Service.category, Service.name)
    )
    services = result.scalars().all()
    return [
        ServiceOut(
            id=svc.id,
            name=svc.name,
            category=svc.category,
            max_members=svc.max_members,
            monthly_price=svc.monthly_price,
            logo_image_url=build_minio_asset_url(svc.logo_image_key),
        )
        for svc in services
    ]

@router.get("/categories", response_model=list[CategoryOut])
async def list_categories(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Service.category).distinct())
    categories = result.scalars().all()
    return [{"name": cat} for cat in categories if cat]  

@router.get("", response_model=PartyListOut)
async def list_parties(
    category_name: Optional[str] = Query(None), 
    service_id: Optional[uuid.UUID] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(12, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    q = select(Party).options(
        selectinload(Party.host),
        selectinload(Party.members),
        selectinload(Party.service),
    )
    if service_id:
        q = q.where(Party.service_id == service_id)
    if category_name:
        q = q.join(Party.service).where(Service.category == category_name) 
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
        size=size,
    )

@router.get("/{party_id}", response_model=PartyOut)
async def get_party(
    party_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    result = await db.execute(
        select(Party)
        .options(selectinload(Party.host), selectinload(Party.members), selectinload(Party.service))
        .where(Party.id == party_id)
    )
    party = result.scalar_one_or_none()
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")
    return _build_party_out(party, current_user.id if current_user else None)

@router.post("", response_model=PartyOut, status_code=status.HTTP_201_CREATED)
async def create_party(
    body: PartyCreate,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    svc = await db.get(Service, body.service_id)
    if not svc:
        raise HTTPException(status_code=404, detail="서비스를 찾을 수 없습니다.")
    if not svc.is_active:
        raise HTTPException(status_code=400, detail="비활성화된 서비스입니다.")

    max_members = body.max_members if body.max_members is not None else svc.max_members

    # 1인당 가격 = (전체요금 / 인원수) * (1 + 수수료율)
    base_per_person = svc.monthly_price / max_members
    commission = svc.commission_rate or 0.0
    monthly_per_person = round(base_per_person * (1 + commission))

    start_date = None
    end_date = None
    if body.start_date:
        try:
            start_date = date.fromisoformat(body.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="start_date 형식 오류 (YYYY-MM-DD)")
    if body.end_date:
        try:
            end_date = date.fromisoformat(body.end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="end_date 형식 오류 (YYYY-MM-DD)")

    party = Party(
        leader_id=current_user.id,
        service_id=body.service_id,
        title=body.title,
        description=body.description,
        max_members=max_members,
        monthly_per_person=monthly_per_person,
        min_trust_score=body.min_trust_score if body.min_trust_score is not None else 0.0,
        status="recruiting",
        start_date=start_date,
        end_date=end_date,
    )
    db.add(party)
    await db.commit()

    result = await db.execute(
        select(Party)
        .options(selectinload(Party.host), selectinload(Party.members), selectinload(Party.service))
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
        select(Party).options(selectinload(Party.service)).where(Party.id == party_id)
    )
    party = result.scalar_one_or_none()
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

    current_count = _party_member_count(party)
    if current_count >= (party.max_members or 0):
        raise HTTPException(status_code=400, detail="파티 인원이 가득 찼습니다.")

    try:
        db.add(PartyMember(party_id=party_id, user_id=current_user.id))
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error joining party: {e}")
        raise HTTPException(status_code=500, detail="파티 가입 처리 중 서버 오류가 발생했습니다.")

    return MessageOut(message="파티 참여가 완료되었습니다.")
