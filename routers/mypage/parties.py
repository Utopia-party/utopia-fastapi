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
    leader_q = select(Party.id).where(Party.leader_id == current_user.id)

    member_q = (
        select(PartyMember.party_id)
        .where(
            PartyMember.user_id == current_user.id,
            PartyMember.status == "active",
        )
    )

    party_id_rows = await db.execute(
        select(Party.id)
        .where(or_(Party.id.in_(leader_q), Party.id.in_(member_q)))
    )
    party_ids: List[uuid.UUID] = [row[0] for row in party_id_rows.all()]

    if not party_ids:
        return MyPartyListOut(parties=[])

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
        has_referrer_discount = False
        if current_user.referrer_id is not None:
            member_user_ids = {m.user_id for m in (p.members or [])}
            has_referrer_discount = current_user.referrer_id in member_user_ids
        items.append(MyPartyOut(
            **base.model_dump(),
            is_owner=(p.leader_id == current_user.id),
            has_referrer_discount=has_referrer_discount,
        ))

    return MyPartyListOut(parties=items)
