"""
마이페이지 — 내 파티 목록 (v2 신규)

- 내가 리더이거나 active 멤버인 파티 전부 반환
- is_owner 플래그로 프론트에서 '내가 만든 파티' 배지 / 리더 전용 버튼 렌더링
"""
import uuid
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database import get_db
from core.security import require_user
from models.party import Party, PartyMember
from models.user import User
from routers.parties import _build_party_out
from schemas.party import MyPartyListOut, MyPartyOut

router = APIRouter(tags=["mypage-parties"])


@router.get("/users/me/parties", response_model=MyPartyListOut)
async def list_my_parties(
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    # 1) 내가 리더인 party_id 수집
    leader_q = select(Party.id).where(Party.leader_id == current_user.id)

    # 2) 내가 active 멤버인 party_id 수집
    member_q = (
        select(PartyMember.party_id)
        .where(
            PartyMember.user_id == current_user.id,
            PartyMember.status == "active",
        )
    )

    # union
    party_id_rows = await db.execute(
        select(Party.id)
        .where(or_(Party.id.in_(leader_q), Party.id.in_(member_q)))
    )
    party_ids: List[uuid.UUID] = [row[0] for row in party_id_rows.all()]

    if not party_ids:
        return MyPartyListOut(parties=[])

    # 3) 해당 파티들 full load
    result = await db.execute(
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.service),
            selectinload(Party.members),
        )
        .where(Party.id.in_(party_ids))
        .order_by(Party.created_at.desc())
    )
    parties = result.scalars().all()

    items: list[MyPartyOut] = []
    for p in parties:
        base = _build_party_out(p, current_user.id)
        items.append(MyPartyOut(
            **base.model_dump(),
            is_owner=(p.leader_id == current_user.id),
        ))

    return MyPartyListOut(parties=items)
