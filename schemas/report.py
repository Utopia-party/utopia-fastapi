from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


ReportTargetType = Literal["USER", "PARTY", "CHAT"]
ReportCategory = Literal["PROFANITY", "SCAM", "SPAM"]
ReportStatus = Literal["PENDING", "IN_REVIEW", "APPROVED", "REJECTED"]


class ReportEvidenceResponse(BaseModel):
    id: UUID
    object_key: str
    original_filename: str | None = None
    content_type: str | None = None
    file_size: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportResponse(BaseModel):
    id: UUID
    reporter_id: UUID
    target_type: ReportTargetType
    target_id: UUID
    target_snapshot_name: str | None = None
    category: str
    description: str
    status: ReportStatus
    action_result_code: str
    admin_memo: str | None = None
    reviewed_by: UUID | None = None
    reviewed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    evidences: list[ReportEvidenceResponse] = []

    model_config = {"from_attributes": True}


class ReportSummaryResponse(BaseModel):
    pending: int = Field(0)
    in_review: int = Field(0)
    approved: int = Field(0)
    rejected: int = Field(0)