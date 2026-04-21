import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.payment import Payment
from models.party import Party


async def get_my_payment_items(
    db: AsyncSession,
    user_id: uuid.UUID,
):
    result = await db.execute(
        select(Payment, Party.title)
        .outerjoin(Party, Payment.party_id == Party.id)
        .where(Payment.user_id == user_id)
        .order_by(Payment.created_at.desc())
    )
    return result.all()