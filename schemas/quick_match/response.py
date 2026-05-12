import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QuickMatchRequestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    service_id: uuid.UUID
    status: str
    retry_count: int
    preferred_conditions: dict[str, Any] | None = None
    matched_party_id: uuid.UUID | None = None
    fail_reason: str | None = None
    request_profile_snapshot: dict[str, Any] | None = None
    requested_at: datetime
    matched_at: datetime | None = None
    expired_at: datetime | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class QuickMatchCandidateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    request_id: uuid.UUID
    party_id: uuid.UUID
    rule_score: float
    probability_score: float
    final_score: float
    rank: int | None = None
    filter_reasons: dict[str, Any] | None = None
    status: str
    created_at: datetime
    updated_at: datetime


class QuickMatchResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    request_id: uuid.UUID
    selected_party_id: uuid.UUID | None = None
    selected_candidate_id: uuid.UUID | None = None
    request_snapshot: dict[str, Any] | None = None
    candidate_snapshot: dict[str, Any] | None = None
    final_scores: dict[str, Any] | None = None
    decision_reason: str | None = None
    created_at: datetime


class QuickMatchCreateResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "빠른매칭 요청이 생성되었습니다.",
                "request_id": "11111111-1111-1111-1111-111111111111",
                "status": "requested",
            }
        }
    )

    message: str = Field(..., description="응답 메시지")
    request_id: uuid.UUID = Field(..., description="생성된 빠른매칭 요청 ID")
    status: str = Field(..., description="요청 상태")


class QuickMatchDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request: QuickMatchRequestResponse
    candidates: list[QuickMatchCandidateResponse] = Field(default_factory=list)
    result: QuickMatchResultResponse | None = None
