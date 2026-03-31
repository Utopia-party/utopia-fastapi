import uuid
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class NotificationOut(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    type: Optional[str]
    title: Optional[str]
    message: Optional[str]
    reference_type: Optional[str]
    reference_id: Optional[uuid.UUID]
    is_read: Optional[bool]
    created_at: Optional[datetime]

    model_config = {"from_attributes": True}