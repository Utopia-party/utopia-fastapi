from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class NotificationOut(BaseModel):
    id: UUID
    user_id: UUID
    is_read: bool
    created_at: datetime
    metadata: dict[str, Any] | None = Field(default=None, alias="meta")
    read_at: datetime | None = None
    reference_id: UUID | None = None
    type: str
    title: str
    message: str
    reference_type: str | None = None

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

# class NotificationResponse(NotificationBase):
#     pass
