import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user_optional
from models.admin import AdminRole, ModerationAction
from models.appeal import BanAppeal
from models.mypage.trust_score import TrustScore
from models.party import PartyMember
from models.refresh_token import RefreshToken
from models.user import User
from schemas.appeal import (
    AdminAppealOut,
    AdminAppealReviewIn,
    AppealCreateIn,
    AppealOut,
)
from services.notifications.appeal_notification_service import (
    notify_admins_new_appeal,
    notify_appeal_result,
    notify_appeal_submitted,
)
from routers.admin.deps import (
    AdminContext,
    require_admin_context,
    _append_activity_log,
)

router = APIRouter(tags=["appeals"])

# ban_type 상수
BAN_TYPE_IP = "ip_ban"
BAN_TYPE_TRUST = "trust_score"
BAN_TYPE_MANUAL = "manual"
BAN_TYPE_REPORT = "report"


def _fmt(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _to_appeal_out(a: BanAppeal) -> AppealOut:
    return AppealOut(
        id=str(a.id),
        user_id=str(a.user_id),
        ban_type=a.ban_type,
        ban_reference_id=str(a.ban_reference_id) if a.ban_reference_id else None,
        reason=a.reason,
        status=a.status,
        admin_memo=a.admin_memo,
        created_at=_fmt(a.created_at),
    )

@router.post("/api/appeals", response_model=AppealOut)
async def create_appeal(
    payload: AppealCreateIn,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    # 정지 유저도 이의제기 가능하지만, 로그인은 필수
    if not current_user:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    ref_id = uuid.UUID(payload.ban_reference_id) if payload.ban_reference_id else None

    if ref_id is not None:
        dup_filter = (
            BanAppeal.user_id == current_user.id,
            BanAppeal.ban_reference_id == ref_id,
            BanAppeal.status.in_(["PENDING", "APPROVED"]),
        )
    else:
        dup_filter = (
            BanAppeal.user_id == current_user.id,
            BanAppeal.ban_type == payload.ban_type,
            BanAppeal.ban_reference_id.is_(None),
            BanAppeal.status.in_(["PENDING", "APPROVED"]),
        )

    existing = await db.scalar(select(BanAppeal).where(*dup_filter))
    if existing:
        raise HTTPException(
            status_code=409,
            detail="이미 해당 제재에 대한 이의제기가 접수되어 있습니다.",
        )

    appeal = BanAppeal(
        user_id=current_user.id,
        ban_type=payload.ban_type,
        ban_reference_id=ref_id,
        reason=payload.reason.strip(),
        status="PENDING",
    )
    db.add(appeal)

    await db.commit()
    await db.refresh(appeal)

    await notify_appeal_submitted(db=db, appeal=appeal)
    await notify_admins_new_appeal(db=db, appeal=appeal, user_nickname=current_user.nickname)

    return _to_appeal_out(appeal)

@router.get("/api/appeals/my", response_model=list[AppealOut])
async def get_my_appeals(
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),  # ← 수정
):
    if not current_user:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    rows = (
        await db.execute(
            select(BanAppeal)
            .where(BanAppeal.user_id == current_user.id)
            .order_by(BanAppeal.created_at.desc())
        )
    ).scalars().all()

    return [_to_appeal_out(r) for r in rows]

# ... (이하 admin 라우터는 동일)
