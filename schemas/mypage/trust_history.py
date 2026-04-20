from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TrustHistoryItemResponse(BaseModel):
    id: str
    title: str
    detail: Optional[str] = None
    score_change: float
    trust_score_after: float
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class MyTrustHistoryResponse(BaseModel):
    items: list[TrustHistoryItemResponse]