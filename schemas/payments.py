import uuid
from datetime import datetime
from pydantic import BaseModel

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