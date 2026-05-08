"""
장기 파티 신뢰도 보너스 태스크
기획서 3.1 가점 로직:
  멤버: 2개월 +1 / 4개월 +3 / 6개월 +5
  방장: 2개월 +2 / 4개월 +4 / 6개월 +6
  파티 시작일(created_at) 기준으로 판단.
매일 00:00 KST에 실행되도록 celery beat에 등록.
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta

from core.celery_app import celery_app
from core.database import AsyncSessionLocal
from models.party import Party, PartyMember
from models.user import User
from models.mypage.trust_score import TrustScore
from sqlalchemy import select


# 기획서 기준 구간 (개월 수, 멤버 보너스, 방장 보너스)
BONUS_TIERS = [
    (6, 5.0, 6.0),
    (4, 3.0, 4.0),
    (2, 1.0, 2.0),
]

# 중복 지급 방지를 위한 reason 
BONUS_REASON_PREFIX = "장기파티보너스"


def _months_elapsed(start: datetime, now: datetime) -> int:
    """start로부터 경과한 완전한 개월 수 반환."""
    delta_days = (now - start).days
    return delta_days // 30


async def _already_granted(db, user_id: uuid.UUID, party_id: uuid.UUID, months: int) -> bool:
    """해당 구간 보너스가 이미 지급됐는지 TrustScore 이력으로 확인."""
    reason_key = f"{BONUS_REASON_PREFIX} {months}M {party_id}"
    result = await db.execute(
        select(TrustScore).where(
            TrustScore.user_id == user_id,
            TrustScore.reason == reason_key,
        )
    )
    return result.scalar_one_or_none() is not None


async def _grant_bonus(
    db,
    user_id: uuid.UUID,
    party_id: uuid.UUID,
    delta: float,
    months: int,
) -> None:
    reason = f"{BONUS_REASON_PREFIX} {months}M {party_id}"
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        return

    previous = float(user.trust_score) if user.trust_score is not None else 36.5
    new_score = min(99.0, round(previous + delta, 1))
    change = round(new_score - previous, 1)
    if change <= 0:
        return

    user.trust_score = new_score
    db.add(
        TrustScore(
            user_id=user_id,
            previous_score=previous,
            new_score=new_score,
            change_amount=change,
            reason=reason,
            created_by=user_id,
        )
    )


async def _run_party_trust_bonus() -> dict:
    now = datetime.now(timezone.utc)
    granted_count = 0

    async with AsyncSessionLocal() as db:
        # ended 상태 제외, 진행 중이거나 완료된 파티만 대상
        party_result = await db.execute(
            select(Party).where(Party.status != "ended")
        )
        parties = party_result.scalars().all()

        for party in parties:
            start = party.created_at
            if start is None:
                continue
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)

            elapsed_months = _months_elapsed(start, now)
            if elapsed_months < 2:
                continue

            # 해당 파티에서 적용할 보너스 구간 결정 (가장 높은 구간 하나만)
            tier = None
            for months, member_bonus, leader_bonus in BONUS_TIERS:
                if elapsed_months >= months:
                    tier = (months, member_bonus, leader_bonus)
                    break
            if tier is None:
                continue

            months, member_bonus, leader_bonus = tier

            # 방장 보너스
            leader_id = party.leader_id
            if leader_id and not await _already_granted(db, leader_id, party.id, months):
                await _grant_bonus(db, leader_id, party.id, leader_bonus, months)
                granted_count += 1

            # 멤버 보너스 (active 상태 멤버만)
            member_result = await db.execute(
                select(PartyMember).where(
                    PartyMember.party_id == party.id,
                    PartyMember.status == "active",
                )
            )
            members = member_result.scalars().all()
            for member in members:
                if member.user_id == leader_id:
                    continue  # 방장은 위에서 처리
                if not await _already_granted(db, member.user_id, party.id, months):
                    await _grant_bonus(db, member.user_id, party.id, member_bonus, months)
                    granted_count += 1

        await db.commit()

    return {"granted_count": granted_count, "run_at": now.isoformat()}


@celery_app.task(
    name="tasks.party_trust_bonus.run_party_trust_bonus",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
)
def run_party_trust_bonus(self):
    try:
        result = asyncio.run(_run_party_trust_bonus())
        return result
    except Exception as exc:
        raise self.retry(exc=exc)
