import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Cookie, Header
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from models.user import User
from services.auth_service import decode_access_token
from services.mypage.payments import get_my_payment_items

router = APIRouter(prefix="/mypage/payments", tags=["mypage-payments"])


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
        user_id_str = payload.get("sub")
        if not user_id_str:
            raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
        user_id = uuid.UUID(user_id_str)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="사용자를 찾을 수 없습니다.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="비활성화된 계정입니다.")

    return user


class MyPaymentItemOut(BaseModel):
    id: uuid.UUID
    party_id: uuid.UUID
    party_title: str | None
    amount: int
    payment_method: str | None
    status: str
    billing_month: str
    paid_at: datetime | None
    created_at: datetime
    pg_transaction_id: str | None

    class Config:
        from_attributes = True


@router.get("", response_model=list[MyPaymentItemOut])
async def get_my_payments(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    print(f"[MYPAGE PAYMENT] /mypage/payments current_user.id={current_user.id}")

    rows = await get_my_payment_items(db=db, user_id=current_user.id)

    print(f"[MYPAGE PAYMENT] rows={len(rows)}")

    return [
        MyPaymentItemOut(
            id=payment.id,
            party_id=payment.party_id,
            party_title=party_title,
            amount=payment.amount,
            payment_method=payment.payment_method,
            status=payment.status,
            billing_month=payment.billing_month,
            paid_at=payment.paid_at,
            created_at=payment.created_at,
            pg_transaction_id=payment.pg_transaction_id,
        )
        for payment, party_title in rows
    ]