import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Cookie, Header
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.payment import Payment
from models.party import Party, PartyMember, Service
from models.user import User
from services.auth_service import decode_access_token
from services.notifications.settlement_notification_service import (
    notify_settlement_requested_to_member,
    notify_member_settlement_completed_to_leader,
)

router = APIRouter(prefix="/payments", tags=["payments"])

async def get_current_user(
    access_token: str | None = Cookie(default=None, alias="access_token"),
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    token = access_token
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    try:
        payload = decode_access_token(token)
        user_id = uuid.UUID(payload.get("sub"))
    except Exception:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="비활성화된 계정입니다.")
    return user

async def _has_referrer_in_party(
    db: AsyncSession,
    party_id: uuid.UUID,
    leader_id: uuid.UUID,
    referrer_id: uuid.UUID | None,
) -> bool:
    if referrer_id is None:
        return False
    if leader_id == referrer_id:
        return True
    result = await db.execute(
        select(PartyMember).where(
            PartyMember.party_id == party_id,
            PartyMember.user_id == referrer_id,
            PartyMember.status == "active",
        )
    )
    return result.scalar_one_or_none() is not None


def _calc_payment(
    base_amount: int,
    service: Service | None,
    is_leader: bool,
    has_referrer: bool,
) -> tuple[int, float, int, str | None]:
    discount_rate = 0.0
    reasons: list[str] = []

    if is_leader and service and service.leader_discount_rate:
        d = float(service.leader_discount_rate)
        discount_rate += d
        reasons.append(f"방장 할인 {int(d * 100)}%")

    if has_referrer and service and service.referral_discount_rate:
        d = float(service.referral_discount_rate)
        discount_rate += d
        reasons.append(f"추천인 할인 {int(d * 100)}%")

    discount_rate = min(discount_rate, 1.0)
    amount = round(base_amount * (1 - discount_rate))

    commission_rate = (
        float(service.commission_rate)
        if service and service.commission_rate is not None
        else 0.30
    )
    commission_amount = round(amount * commission_rate / (1 + commission_rate)) if commission_rate > 0 else 0

    discount_reason = " + ".join(reasons) if reasons else None
    return amount, commission_rate, commission_amount, discount_reason


# ── 포트원 검증 ──────────────────────────────────────────────────

async def _verify_portone(payment_id: str, expected_amount: int) -> None:
    from core.config import settings
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://api.portone.io/payments/{payment_id}",
                headers={"Authorization": f"PortOne {settings.PORTONE_API_SECRET}"},
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"포트원 결제 조회 실패: {resp.text[:100]}")
        data = resp.json()
        if data.get("status") != "PAID":
            raise HTTPException(status_code=400, detail=f"결제 미완료 상태: {data.get('status')}")
        if data.get("amount", {}).get("total") != expected_amount:
            raise HTTPException(status_code=400, detail="결제 금액 불일치")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"포트원 검증 중 오류: {e}")


# ── 스키마 ───────────────────────────────────────────────────────

class CardConfirmRequest(BaseModel):
    party_id: uuid.UUID
    pg_transaction_id: str
    amount: int


class TransferRegisterRequest(BaseModel):
    party_id: uuid.UUID
    amount: int


class PaymentOut(BaseModel):
    id: uuid.UUID
    status: str
    payment_method: str | None
    amount: int
    billing_month: str
    paid_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


# ── 엔드포인트 ───────────────────────────────────────────────────

@router.get("/status")
async def get_payment_status(
    party_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    billing_month = datetime.now(timezone.utc).strftime("%Y-%m")
    result = await db.execute(
        select(Payment).where(
            Payment.user_id == current_user.id,
            Payment.party_id == party_id,
            Payment.billing_month == billing_month,
            Payment.status == "approved",
        )
    )
    return {"paid": result.scalar_one_or_none() is not None, "billing_month": billing_month}


@router.post("/card/confirm", response_model=PaymentOut)
async def card_confirm(
    body: CardConfirmRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(
        select(Payment).where(Payment.pg_transaction_id == body.pg_transaction_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="이미 처리된 결제입니다.")

    now = datetime.now(timezone.utc)
    billing_month = now.strftime("%Y-%m")

    duplicate = await db.execute(
        select(Payment).where(
            Payment.user_id == current_user.id,
            Payment.party_id == body.party_id,
            Payment.billing_month == billing_month,
            Payment.status == "approved",
        )
    )
    if duplicate.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="이번 달 결제가 이미 완료되었습니다.")

    party = await db.get(Party, body.party_id)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    service = await db.get(Service, party.service_id) if party.service_id else None
    is_leader = party.leader_id == current_user.id
    has_referrer = await _has_referrer_in_party(
        db, body.party_id, party.leader_id, current_user.referrer_id
    )

    base_price = int(party.monthly_per_person) if party.monthly_per_person else 0
    amount, commission_rate, commission_amount, discount_reason = _calc_payment(
        base_price, service, is_leader, has_referrer
    )

    await _verify_portone(body.pg_transaction_id, amount)

    payment = Payment(
        user_id=current_user.id,
        party_id=body.party_id,
        base_price=base_price,
        commission_rate=commission_rate,
        commission_amount=commission_amount,
        discount_reason=discount_reason,
        amount=amount,
        payment_method="card",
        status="approved",
        billing_month=billing_month,
        paid_at=now,
        pricing_type="normal",
        pg_provider="portone",
        pg_transaction_id=body.pg_transaction_id,
    )
    db.add(payment)
    await db.commit()
    await db.refresh(payment)

    await notify_member_settlement_completed_to_leader(
        db=db,
        party=party,
        member_user_id=current_user.id,
        member_nickname=current_user.nickname,
    )
    return payment


@router.post("/transfer/register", response_model=PaymentOut)
async def transfer_register(
    body: TransferRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    billing_month = now.strftime("%Y-%m")

    duplicate = await db.execute(
        select(Payment).where(
            Payment.user_id == current_user.id,
            Payment.party_id == body.party_id,
            Payment.billing_month == billing_month,
            Payment.status.in_(["approved", "pending"]),
        )
    )
    if duplicate.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="이번 달 결제가 이미 존재합니다.")

    party = await db.get(Party, body.party_id)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    service = await db.get(Service, party.service_id) if party.service_id else None
    is_leader = party.leader_id == current_user.id
    has_referrer = await _has_referrer_in_party(
        db, body.party_id, party.leader_id, current_user.referrer_id
    )

    base_price = int(party.monthly_per_person) if party.monthly_per_person else 0
    amount, commission_rate, commission_amount, discount_reason = _calc_payment(
        base_price, service, is_leader, has_referrer
    )

    payment = Payment(
        user_id=current_user.id,
        party_id=body.party_id,
        base_price=base_price,
        commission_rate=commission_rate,
        commission_amount=commission_amount,
        discount_reason=discount_reason,
        amount=amount,
        payment_method="transfer",
        status="pending",
        billing_month=billing_month,
        paid_at=None,
        pricing_type="normal",
        pg_provider=None,
        pg_transaction_id=None,
    )
    db.add(payment)
    await db.commit()
    await db.refresh(payment)

    if current_user.id != party.leader_id:
        await notify_settlement_requested_to_member(
            db=db,
            party=party,
            member_user_id=party.leader_id,
            amount=amount,
        )
    return payment


@router.patch("/{payment_id}/approve")
async def approve_transfer(
    payment_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="관리자만 접근 가능합니다.")

    payment = await db.get(Payment, payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="결제 내역을 찾을 수 없습니다.")
    if payment.status != "pending":
        raise HTTPException(status_code=400, detail="대기 중인 결제가 아닙니다.")

    payment.status = "approved"
    payment.paid_at = datetime.now(timezone.utc)
    await db.commit()

    user = await db.get(User, payment.user_id)
    party = await db.get(Party, payment.party_id)
    await notify_member_settlement_completed_to_leader(
        db=db,
        party=party,
        member_user_id=payment.user_id,
        member_nickname=user.nickname if user else None,
    )
    return {"message": "승인 완료", "payment_id": str(payment_id)}
