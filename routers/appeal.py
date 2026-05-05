"""
이의제기 라우터
- POST   /api/appeals                유저: 이의제기 신청 (정지 상태에서도 가능)
- GET    /api/appeals/my             유저: 내 이의제기 목록
- GET    /api/admin/appeals          관리자: 전체 목록
- PATCH  /api/admin/appeals/{id}     관리자: 승인/거부
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user
from models.admin import AdminRole, ModerationAction
from models.appeal import BanAppeal
from models.mypage.trust_score import TrustScore
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


# ── 유저: 이의제기 신청 ──────────────────────────────────────────────────────
@router.post("/api/appeals", response_model=AppealOut)
async def create_appeal(
    payload: AppealCreateIn,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")

    ref_id = uuid.UUID(payload.ban_reference_id) if payload.ban_reference_id else None

    # 중복 차단
    if ref_id is not None:
        # reference_id 있으면 동일 제재 건 기준
        dup_filter = (
            BanAppeal.user_id == current_user.id,
            BanAppeal.ban_reference_id == ref_id,
            BanAppeal.status.in_(["PENDING", "APPROVED"]),
        )
    else:
        # ip_ban / manual 등 reference_id 없는 경우 → ban_type 기준
        # (NULL = NULL은 DB에서 false라 코드 레벨에서 체크)
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

    # appeal 먼저 commit 후 알림 발송 (트랜잭션 순서 보장)
    await db.commit()
    await db.refresh(appeal)

    await notify_appeal_submitted(db=db, appeal=appeal)
    await notify_admins_new_appeal(db=db, appeal=appeal, user_nickname=current_user.nickname)

    return _to_appeal_out(appeal)


# ── 유저: 내 이의제기 목록 ────────────────────────────────────────────────────
@router.get("/api/appeals/my", response_model=list[AppealOut])
async def get_my_appeals(
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
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


# ── 관리자: 이의제기 목록 ────────────────────────────────────────────────────
@router.get("/api/admin/appeals", response_model=list[AdminAppealOut])
async def get_admin_appeals(
    status_filter: str = Query(default="", alias="status"),
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(BanAppeal).order_by(BanAppeal.created_at.desc())
    if status_filter:
        stmt = stmt.where(BanAppeal.status == status_filter.upper())

    appeals = (await db.execute(stmt)).scalars().all()

    if not appeals:
        return []

    # 관련 유저 일괄 로드
    user_ids = {a.user_id for a in appeals}
    reviewer_ids = {a.reviewed_by for a in appeals if a.reviewed_by}
    all_ids = user_ids | reviewer_ids
    users: dict = {}
    if all_ids:
        users = {
            u.id: u
            for u in (await db.execute(select(User).where(User.id.in_(all_ids)))).scalars().all()
        }

    # 관련 제재 기록 일괄 로드
    ref_ids = {a.ban_reference_id for a in appeals if a.ban_reference_id}
    moderation_map: dict = {}
    trust_map: dict = {}
    if ref_ids:
        mods = (
            await db.execute(select(ModerationAction).where(ModerationAction.id.in_(ref_ids)))
        ).scalars().all()
        moderation_map = {m.id: m for m in mods}

        trusts = (
            await db.execute(select(TrustScore).where(TrustScore.id.in_(ref_ids)))
        ).scalars().all()
        trust_map = {t.id: t for t in trusts}

    result = []
    for a in appeals:
        user = users.get(a.user_id)
        reviewer = users.get(a.reviewed_by) if a.reviewed_by else None

        ban_detail = None
        ban_score_change = None
        ban_created_at = None

        if a.ban_reference_id:
            mod = moderation_map.get(a.ban_reference_id)
            trust = trust_map.get(a.ban_reference_id)
            if mod:
                ban_detail = mod.reason
                ban_score_change = float(mod.trust_score_change) if mod.trust_score_change else None
                ban_created_at = _fmt(mod.created_at)
            elif trust:
                ban_detail = trust.reason
                ban_score_change = float(trust.change_amount)
                ban_created_at = _fmt(trust.created_at)

        result.append(
            AdminAppealOut(
                id=str(a.id),
                user_id=str(a.user_id),
                user_nickname=user.nickname if user else "알 수 없음",
                user_email=user.email if user else "",
                ban_type=a.ban_type,
                ban_reference_id=str(a.ban_reference_id) if a.ban_reference_id else None,
                reason=a.reason,
                status=a.status,
                admin_memo=a.admin_memo,
                reviewed_by_nickname=reviewer.nickname if reviewer else None,
                reviewed_at=_fmt(a.reviewed_at),
                created_at=_fmt(a.created_at),
                ban_detail=ban_detail,
                ban_score_change=ban_score_change,
                ban_created_at=ban_created_at,
            )
        )

    return result


# ── 관리자: 이의제기 처리 (승인/거부) ────────────────────────────────────────
@router.patch("/api/admin/appeals/{appeal_id}", response_model=AdminAppealOut)
async def review_appeal(
    appeal_id: str,
    payload: AdminAppealReviewIn,
    admin: AdminContext = Depends(require_admin_context),
    db: AsyncSession = Depends(get_db),
):
    try:
        aid = uuid.UUID(appeal_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="유효하지 않은 ID입니다.")

    appeal = await db.get(BanAppeal, aid)
    if not appeal:
        raise HTTPException(status_code=404, detail="이의제기를 찾을 수 없습니다.")
    if appeal.status != "PENDING":
        raise HTTPException(status_code=409, detail="이미 처리된 이의제기입니다.")

    status_upper = payload.status.upper()
    if status_upper not in {"APPROVED", "REJECTED"}:
        raise HTTPException(status_code=400, detail="status는 APPROVED 또는 REJECTED여야 합니다.")

    appeal.status = status_upper
    appeal.reviewed_by = admin.user.id
    appeal.reviewed_at = datetime.now(timezone.utc)
    appeal.admin_memo = payload.admin_memo

    target_user = await db.get(User, appeal.user_id)

    if status_upper == "APPROVED" and target_user:
        # 1. 계정 정지 해제
        target_user.is_active = True
        target_user.banned_until = None

        # 2. IP 밴 해제 (Redis)
        if appeal.ban_type == BAN_TYPE_IP:
            from core.redis_client import redis_client
            token_row = await db.scalar(
                select(RefreshToken)
                .where(RefreshToken.user_id == target_user.id)
                .order_by(RefreshToken.created_at.desc())
                .limit(1)
            )
            if token_row and token_row.ip_address:
                await redis_client.delete(f"ip:banned:{token_row.ip_address}")

        # 3. trust_score 복구 (잃은 만큼)
        if appeal.ban_reference_id:
            trust_row = await db.get(TrustScore, appeal.ban_reference_id)
            if trust_row and float(trust_row.change_amount) < 0:
                recovery = abs(float(trust_row.change_amount))
                prev = float(target_user.trust_score)
                new_score = min(round(prev + recovery, 1), 99.0)
                target_user.trust_score = new_score

                db.add(TrustScore(
                    user_id=target_user.id,
                    previous_score=prev,
                    new_score=new_score,
                    change_amount=round(new_score - prev, 1),
                    reason="이의제기 승인 - 점수 복구",
                    created_by=admin.user.id,
                    reference_id=appeal.id,
                ))

        await _append_activity_log(
            db,
            actor_user_id=admin.user.id,
            action_type="APPEAL_APPROVED",
            description=f"{target_user.nickname} 이의제기 승인 - 제재 해제",
            path=f"/api/admin/appeals/{appeal_id}",
            target_id=target_user.id,
        )
    else:
        await _append_activity_log(
            db,
            actor_user_id=admin.user.id,
            action_type="APPEAL_REJECTED",
            description=f"{target_user.nickname if target_user else appeal_id} 이의제기 거부",
            path=f"/api/admin/appeals/{appeal_id}",
            target_id=target_user.id if target_user else None,
        )

    # appeal 먼저 commit 후 알림 발송 (트랜잭션 순서 보장)
    await db.commit()
    await db.refresh(appeal)

    await notify_appeal_result(db=db, appeal=appeal)

    # 해당 건만 직접 조회해서 반환 (전체 재조회 성능 문제 해결)
    return await _get_single_admin_appeal(aid, db)


async def _get_single_admin_appeal(appeal_id: uuid.UUID, db: AsyncSession) -> AdminAppealOut:
    appeal = await db.get(BanAppeal, appeal_id)
    if not appeal:
        raise HTTPException(status_code=404, detail="이의제기를 찾을 수 없습니다.")

    user = await db.get(User, appeal.user_id)
    reviewer = await db.get(User, appeal.reviewed_by) if appeal.reviewed_by else None

    ban_detail = None
    ban_score_change = None
    ban_created_at = None

    if appeal.ban_reference_id:
        mod = await db.get(ModerationAction, appeal.ban_reference_id)
        trust = await db.get(TrustScore, appeal.ban_reference_id)
        if mod:
            ban_detail = mod.reason
            ban_score_change = float(mod.trust_score_change) if mod.trust_score_change else None
            ban_created_at = _fmt(mod.created_at)
        elif trust:
            ban_detail = trust.reason
            ban_score_change = float(trust.change_amount)
            ban_created_at = _fmt(trust.created_at)

    return AdminAppealOut(
        id=str(appeal.id),
        user_id=str(appeal.user_id),
        user_nickname=user.nickname if user else "알 수 없음",
        user_email=user.email if user else "",
        ban_type=appeal.ban_type,
        ban_reference_id=str(appeal.ban_reference_id) if appeal.ban_reference_id else None,
        reason=appeal.reason,
        status=appeal.status,
        admin_memo=appeal.admin_memo,
        reviewed_by_nickname=reviewer.nickname if reviewer else None,
        reviewed_at=_fmt(appeal.reviewed_at),
        created_at=_fmt(appeal.created_at),
        ban_detail=ban_detail,
        ban_score_change=ban_score_change,
        ban_created_at=ban_created_at,
    )
