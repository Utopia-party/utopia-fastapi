import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from core.database import get_db
from core.config import settings
from models.payment import Payment
from models.party import Party, PartyMember
import httpx

router = APIRouter(prefix="/payments", tags=["payments"])

PORTONE_API_SECRET = settings.PORTONE_API_SECRET  # 환경변수에 추가 필요


# ── 스키마 ──────────────────────────────────────────────────

class CardConfirmRequest(BaseModel):
    party_id: uuid.UUID
    pg_transaction_id: str  # 포트원 paymentId
    amount: int

class TransferRegisterRequest(BaseModel):
    party_id: uuid.UUID
    amount: int

class PaymentOut(BaseModel):
    id: uuid.UUID
    status: str
    payment_method: str
    amount: int
    billing_month: str
    paid_at: datetime | None
    created_at: datetime


# ── 포트원 결제 검증 ──────────────────────────────────────────

async def verify_portone_payment(payment_id: str, expected_amount: int) -> dict:
    """포트원 서버에서 결제 상태 검증"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.portone.io/payments/{payment_id}",
            headers={"Authorization": f"PortOne {PORTONE_API_SECRET}"},
        )
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail="포트원 결제 조회 실패")

    data = resp.json()
    if data.get("status") != "PAID":
        raise HTTPException(status_code=400, detail="결제가 완료되지 않았습니다.")
    if data.get("amount", {}).get("total") != expected_amount:
        raise HTTPException(status_code=400, detail="결제 금액 불일치")
    return data


# ── 현재 로그인 유저 가져오기 (기존 auth 방식에 맞게) ────────────

from routers.auth import get_current_user  # 기존 auth 라우터의 의존성 재사용
from models.user import User


# ── 카드 결제 승인 ────────────────────────────────────────────

@router.post("/card/confirm", response_model=PaymentOut)
async def card_confirm(
    body: CardConfirmRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 1. 포트원 서버에서 결제 검증
    portone_data = await verify_portone_payment(body.pg_transaction_id, body.amount)

    # 2. 중복 결제 방지 (같은 pg_transaction_id 존재하면 거부)
    existing = await db.execute(
        select(Payment).where(Payment.pg_transaction_id == body.pg_transaction_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="이미 처리된 결제입니다.")

    # 3. 파티 정보 조회
    party = await db.get(Party, body.party_id)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    # 4. 결제 레코드 생성 (바로 approved)
    now = datetime.now(timezone.utc)
    billing_month = now.strftime("%Y-%m")
    commission_rate = 0.10
    commission_amount = int(body.amount * commission_rate)

    payment = Payment(
        user_id=current_user.id,
        party_id=body.party_id,
        base_price=body.amount,
        commission_rate=commission_rate,
        commission_amount=commission_amount,
        amount=body.amount,
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
    return payment


# ── 무통장입금 등록 (pending) ─────────────────────────────────

@router.post("/transfer/register", response_model=PaymentOut)
async def transfer_register(
    body: TransferRegisterRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # 파티 존재 확인
    party = await db.get(Party, body.party_id)
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    now = datetime.now(timezone.utc)
    billing_month = now.strftime("%Y-%m")
    commission_rate = 0.10
    commission_amount = int(body.amount * commission_rate)

    payment = Payment(
        user_id=current_user.id,
        party_id=body.party_id,
        base_price=body.amount,
        commission_rate=commission_rate,
        commission_amount=commission_amount,
        amount=body.amount,
        payment_method="transfer",
        status="pending",   # 관리자 승인 대기
        billing_month=billing_month,
        paid_at=None,
        pricing_type="normal",
        pg_provider=None,
        pg_transaction_id=None,
    )
    db.add(payment)
    await db.commit()
    await db.refresh(payment)
    return payment


# ── 관리자용: 무통장 승인 ─────────────────────────────────────

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
    return {"message": "승인 완료", "payment_id": str(payment_id)}
