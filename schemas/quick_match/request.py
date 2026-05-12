import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuickMatchCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "service_id": "11111111-1111-1111-1111-111111111111",
                "preferred_conditions": {
                    "duration_preference": "over_3_months",
                },
            }
        }
    )

    service_id: uuid.UUID = Field(..., description="서비스 ID")
    preferred_conditions: dict[str, Any] | None = Field(
        default=None,
        description="빠른매칭 선호 조건 JSON",
    )

    @field_validator("preferred_conditions")
    @classmethod
    def normalize_preferred_conditions(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None

        normalized = dict(value)

        duration_preference = normalized.get("duration_preference")
        if isinstance(duration_preference, str):
            normalized["duration_preference"] = duration_preference.strip().lower()

        return normalized


class QuickMatchRetryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reason": "manual_retry",
            }
        }
    )

    reason: str | None = Field(
        default="manual_retry",
        max_length=255,
        description="재탐색 요청 사유",
    )

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        return value or "manual_retry"
