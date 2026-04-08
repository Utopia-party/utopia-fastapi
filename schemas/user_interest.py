from pydantic import BaseModel, Field


class UserInterestUpdateRequest(BaseModel):
    # service_id UUID 문자열 배열
    items: list[str] = Field(default_factory=list)


class UserInterestListResponse(BaseModel):
    # 저장된 service_id UUID 문자열 배열 반환
    items: list[str]

