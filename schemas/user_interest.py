from pydantic import BaseModel, Field


# 상원: 이 클래스는 프론트가 서버로 보낼 관심사 목록 요청 바디 형식을 정의합니다.
class UserInterestUpdateRequest(BaseModel):
    # 상원: items 필드는 사용자가 고른 관심사 문자열 배열을 그대로 담습니다.
    items: list[str] = Field(default_factory=list)


# 상원: 이 클래스는 서버가 프론트로 돌려줄 관심사 목록 응답 형식을 정의합니다.
class UserInterestListResponse(BaseModel):
    # 상원: 저장 후나 조회 시 최종 확정된 관심사 목록을 items 배열로 반환합니다.
    items: list[str]
