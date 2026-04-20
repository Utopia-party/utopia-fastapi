from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.mypage.trust_score import TrustScore
from schemas.mypage.trust_history import (
    MyTrustHistoryResponse,
    TrustHistoryItemResponse,
)


def _build_detail(row: TrustScore) -> str | None:
    parts: list[str] = []

    if row.previous_score is not None and row.new_score is not None:
        parts.append(
            f"{float(row.previous_score):.1f} → {float(row.new_score):.1f}"
        )

    if row.reference_id:
        parts.append(f"reference_id: {row.reference_id}")

    if not parts:
        return None

    return " | ".join(parts)


async def get_my_trust_history_service(
    db: AsyncSession,
    current_user: User,
) -> MyTrustHistoryResponse:
    stmt = (
        select(TrustScore)
        .where(TrustScore.user_id == current_user.id)
        .order_by(desc(TrustScore.created_at), desc(TrustScore.id))
    )

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return MyTrustHistoryResponse(
        items=[
            TrustHistoryItemResponse(
                id=str(row.id),
                title=row.reason,
                detail=_build_detail(row),
                score_change=float(row.change_amount),
                trust_score_after=float(row.new_score),
                created_at=row.created_at,
            )
            for row in rows
        ]
    )