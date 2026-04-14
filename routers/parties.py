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

# --- 계산 유틸리티 함수 추가 ---

def _service_monthly_price(service: Service | None) -> int | None:
    if service is None:
        return None
    return service.monthly_price

def _party_max_members(party: Party, service: Service | None) -> int | None:
    return party.max_members or (service.max_members if service else None)

def _party_member_count(party: Party) -> int:
    # DB에 저장된 current_members가 있으면 우선 사용
    if party.current_members is not None:
        return party.current_members
    # 없으면 연관된 members 리스트로 계산
    member_count = len(party.members) if party.members is not None else 0
    # 방장(leader)을 포함하지 않은 리스트라면 +1을 고려해야 할 수 있으나, 
    # 보통 PartyMember에 방장도 포함되는 설계라면 위 코드로 충분합니다.
    return member_count

def _party_total_price(party: Party, service: Service | None) -> int | None:
    max_members = _party_max_members(party, service)
    if party.monthly_per_person is not None and max_members:
        return party.monthly_per_person * max_members
    return _service_monthly_price(service)

def _service_original_price(service: Service | None) -> int | None:
    if service is None:
        return None
    return service.original_price if service.original_price is not None else service.monthly_price

# --- API 내부 로직 수정 ---

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
        max_members=_party_max_members(party, svc), # 유틸리티 사용
        monthly_price=_service_monthly_price(svc),   # 유틸리티 사용
        logo_image_key=svc.logo_image_key if svc else None,
        logo_image_url=build_minio_asset_url(svc.logo_image_key) if svc else None,
        member_count=_party_member_count(party),    # 유틸리티 사용
        is_joined=is_joined,
    )

# ... (list_services, list_categories, list_parties, get_party 함수들은 동일) ...

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

    # max_members: 요청값 우선, 없으면 서비스 기본값
    max_members = body.max_members if body.max_members is not None else svc.max_members
    # monthly_per_person: 항상 서비스 값 사용 (고정)
    monthly_per_person = svc.monthly_price

    start_date_obj = None
    end_date_obj = None
    if body.start_date:
        try:
            start_date_obj = date.fromisoformat(body.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="start_date 형식 오류 (YYYY-MM-DD)")
    if body.end_date:
        try:
            end_date_obj = date.fromisoformat(body.end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="end_date 형식 오류 (YYYY-MM-DD)")

    party = Party(
        leader_id=current_user.id,
        service_id=body.service_id,
        title=body.title,
        description=body.description,
        max_members=max_members,
        current_members=1, # 생성 시 방장 1명
        monthly_per_person=monthly_per_person,
        min_trust_score=body.min_trust_score if body.min_trust_score is not None else 0.0,
        status="recruiting",
        start_date=start_date_obj,
        end_date=end_date_obj,
    )
    db.add(party)
    
    # 방장을 멤버 테이블에도 추가 (일관성을 위해)
    db.add(PartyMember(party=party, user_id=current_user.id, role="leader"))
    
    await db.commit()

    # 결과 반환을 위한 다시 읽기
    result = await db.execute(
        select(Party)
        .options(selectinload(Party.host), selectinload(Party.members), selectinload(Party.service))
        .where(Party.id == party.id)
    )
    return _build_party_out(result.scalar_one(), current_user.id)

# ... (join_party 함수 생략) ...
