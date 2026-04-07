from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import require_user
from models.user import User
from models.user_interest import UserInterest
from schemas.user_interest import UserInterestListResponse, UserInterestUpdateRequest

router = APIRouter(tags=["user-interests"])


# 상원: 이 함수는 프론트에서 넘어온 관심사 배열을 저장 가능한 형태로 정리합니다.
def normalize_interest_items(items: list[str]) -> list[str]:
    # 상원: 최종 저장할 관심사 배열을 담을 빈 리스트를 먼저 만듭니다.
    normalized: list[str] = []
    # 상원: 이미 본 관심사 이름을 기억해서 중복을 제거하려고 집합을 만듭니다.
    seen: set[str] = set()

    # 상원: 사용자가 보낸 관심사 하나씩을 순서대로 검사합니다.
    for item in items:
        # 상원: 앞뒤 공백을 제거한 뒤 실제 저장 후보 문자열을 만듭니다.
        cleaned = item.strip()
        # 상원: 빈 문자열이거나 너무 긴 값이면 저장하지 않고 건너뜁니다.
        if not cleaned or len(cleaned) > 100:
            continue
        # 상원: 이미 같은 관심사를 본 적이 있으면 중복이므로 건너뜁니다.
        if cleaned in seen:
            continue
        # 상원: 이번 관심사는 이미 처리한 값 목록에 등록합니다.
        seen.add(cleaned)
        # 상원: 정제된 관심사를 최종 저장 배열 뒤에 추가합니다.
        normalized.append(cleaned)

    # 상원: 공백 제거와 중복 제거가 끝난 배열을 호출한 곳으로 돌려줍니다.
    return normalized


@router.get("/users/me/interests", response_model=UserInterestListResponse)
async def get_my_interests(
    # 상원: 현재 로그인한 사용자를 JWT/쿠키 기반으로 주입받습니다.
    current_user: User = Depends(require_user),
    # 상원: 비동기 DB 세션을 FastAPI 의존성으로 주입받습니다.
    db: AsyncSession = Depends(get_db),
):
    # 상원: 현재 사용자 id에 해당하는 관심사만 sort_order 순서대로 조회합니다.
    result = await db.execute(
        select(UserInterest)
        .where(UserInterest.user_id == current_user.id)
        .order_by(UserInterest.sort_order.asc(), UserInterest.created_at.asc())
    )
    # 상원: ORM 객체 리스트에서 프론트가 바로 쓸 interest_name 문자열만 뽑아냅니다.
    items = [interest.interest_name for interest in result.scalars().all()]
    # 상원: 조회한 문자열 배열을 응답 스키마에 담아 반환합니다.
    return UserInterestListResponse(items=items)


@router.put("/users/me/interests", response_model=UserInterestListResponse)
async def update_my_interests(
    # 상원: 프론트가 보낸 관심사 배열 요청 바디를 payload로 받습니다.
    payload: UserInterestUpdateRequest,
    # 상원: 어떤 사용자의 관심사를 바꿀지 현재 로그인 사용자 정보를 주입받습니다.
    current_user: User = Depends(require_user),
    # 상원: 삭제와 재삽입을 수행할 DB 세션을 받습니다.
    db: AsyncSession = Depends(get_db),
):
    # 상원: 공백 제거와 중복 제거를 거친 최종 관심사 배열을 만듭니다.
    items = normalize_interest_items(payload.items)

    # 상원: 기존 관심사는 전부 지워서 현재 선택 상태 전체가 최신본이 되도록 만듭니다.
    await db.execute(delete(UserInterest).where(UserInterest.user_id == current_user.id))

    # 상원: 정리된 관심사를 순서와 함께 하나씩 다시 삽입합니다.
    for index, item in enumerate(items):
        db.add(
            UserInterest(
                # 상원: 현재 로그인 사용자 id를 user_id 컬럼에 넣습니다.
                user_id=current_user.id,
                # 상원: 이번 항목의 실제 관심사 이름을 저장합니다.
                interest_name=item,
                # 상원: 프론트에서 선택한 순서를 유지하도록 index를 sort_order에 넣습니다.
                sort_order=index,
            )
        )

    # 상원: 삭제와 재삽입 결과를 실제 DB에 반영합니다.
    await db.commit()
    # 상원: 저장이 끝난 뒤 프론트가 바로 같은 배열을 다시 사용할 수 있도록 응답으로 돌려줍니다.
    return UserInterestListResponse(items=items)
