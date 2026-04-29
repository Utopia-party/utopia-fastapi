import uuid
import logging
from datetime import date
from typing import Optional, Any

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.config import settings
from core.database import get_db
from core.minio_assets import build_minio_asset_url
from core.security import require_user, get_current_user_optional
from models.admin import ActivityLog
from models.party import Party, PartyMember, Service
from models.user import User
from schemas.party import (
    CategoryOut,
    PartyCreate,
    PartyListOut,
    PartyMembersOut,
    PartyMemberOut,
    PartyOut,
    ServiceOut,
    TransferLeaderRequest,
)
from schemas.user import MessageOut

from services.notifications.party_notification_service import (
    notify_party_join_approved,
    notify_party_join_requested,
    notify_party_join_request_rejected,
    notify_party_join_request_submitted,
    notify_party_member_joined_to_leader,
    notify_party_member_kicked,
)

router = APIRouter(prefix="/parties", tags=["parties"])
logger = logging.getLogger(__name__)

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


async def _broadcast_party_updated(party_id: uuid.UUID) -> None:
    try:
        from routers.chat import manager
        from datetime import datetime, timezone
        await manager.broadcast(str(party_id), {
            "type": "party_updated",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        logger.warning(f"[party_updated broadcast failed] {e}")


async def create_activity_log(
    db: AsyncSession,
    user_id: uuid.UUID,
    action: str,
    description: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    target_id: Optional[uuid.UUID] = None,
) -> None:
    log = ActivityLog(
        actor_user_id=user_id,
        action_type=action,
        description=description,
        extra_metadata=metadata or {},
        target_id=target_id,
    )
    db.add(log)


def _service_monthly_price(service: Service | None) -> int | None:
    if service is None:
        return None
    return service.monthly_price


def _party_max_members(party: Party, service: Service | None) -> int | None:
    return party.max_members or (service.max_members if service else None)

def _party_member_count(party: Party) -> int:
    # 항상 실제 members 관계로 계산 (current_members 캐시 미신뢰)
    # 방장도 PartyMember에 role=leader로 포함되므로 +1 없이 COUNT
    return sum(
        1 for m in (party.members or [])
        if (m.status or "active").lower() == "active"
    )


async def _sync_member_count(db: AsyncSession, party: Party) -> int:
    """PartyMember 테이블 기준으로 current_members 재계산 후 동기화
    방장도 PartyMember에 role=leader로 포함되어 있으므로 +1 없이 COUNT만"""
    result = await db.execute(
        select(func.count(PartyMember.user_id))
        .where(PartyMember.party_id == party.id, PartyMember.status == "active")
    )
    real_count = result.scalar() or 0
    party.current_members = real_count
    return real_count


def _service_original_price(service: Service | None) -> int | None:
    if service is None:
        return None
    return service.original_price


def _build_party_out(
    party: Party,
    current_user_id: Optional[uuid.UUID] = None,
) -> PartyOut:
    svc = party.service
    is_joined = False
    my_member_status: Optional[str] = None

    if current_user_id:
        is_leader = party.leader_id == current_user_id
        my_row = next(
            (m for m in (party.members or []) if m.user_id == current_user_id),
            None,
        )
        my_row_status = (my_row.status or "").lower() if my_row else None
        is_member_active = my_row_status == "active"
        is_joined = is_leader or is_member_active

        if is_leader:
            my_member_status = "leader"
        elif my_row_status:
            my_member_status = my_row_status

    return PartyOut(
        id=party.id,
        leader_id=party.leader_id,
        service_id=party.service_id,
        title=party.title,
        status=party.status,
        host_nickname=party.host.nickname if party.host else None,
        host_trust_score=float(party.host.trust_score) if party.host and party.host.trust_score is not None else None,
        service_name=svc.name if svc else None,
        category_name=svc.category if svc else None,
        max_members=_party_max_members(party, svc),
        monthly_price=party.monthly_per_person,
        original_price=_service_original_price(svc),
        service_total_price=svc.monthly_price if svc else None,
        member_count=_party_member_count(party),
        logo_image_key=svc.logo_image_key if svc else None,
        logo_image_url=build_minio_asset_url(svc.logo_image_key) if svc else None,
        is_joined=is_joined,
        my_member_status=my_member_status,
        start_date=party.start_date,
        end_date=party.end_date,
        min_trust_score=party.min_trust_score,
        created_at=party.created_at,
    )


async def consume_captcha_pass_token(pass_token: str) -> None:
    if not pass_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="캡챠 인증이 필요합니다.",
        )

    redis_key = f"captcha_pass:{pass_token}"

    try:
        token_value = await redis_client.getdel(redis_key)
    except AttributeError:
        token_value = await redis_client.get(redis_key)
        if token_value:
            await redis_client.delete(redis_key)

    if not token_value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="캡챠 인증이 만료되었거나 유효하지 않습니다. 다시 인증해주세요.",
        )


@router.get("/services", response_model=list[ServiceOut])
async def list_services(db: AsyncSession = Depends(get_db)):
    category_order = case(
        (Service.category == "OTT", 1),
        (Service.category == "교육/도서", 2),
        (Service.category == "음악/멤버십", 3),
        (Service.category == "생산성/기타", 4),
        else_=5,
    )

    result = await db.execute(
        select(Service)
        .where(Service.is_active.is_(True))
        .order_by(category_order, Service.name)
    )
    services = result.scalars().all()

    return [
        ServiceOut(
            id=svc.id,
            name=svc.name,
            category=svc.category,
            max_members=svc.max_members,
            monthly_price=svc.monthly_price,
            logo_image_url=build_minio_asset_url(svc.logo_image_key),
        )
        for svc in services
    ]


@router.get("/categories", response_model=list[CategoryOut])
async def list_categories(db: AsyncSession = Depends(get_db)):
    category_order = case(
        (Service.category == "OTT", 1),
        (Service.category == "교육/도서", 2),
        (Service.category == "음악/멤버십", 3),
        (Service.category == "생산성/기타", 4),
        else_=5,
    )

    result = await db.execute(
        select(Service.category)
        .where(Service.category.is_not(None))
        .group_by(Service.category)
        .order_by(category_order, Service.category)
    )

    categories = result.scalars().all()
    return [{"name": cat} for cat in categories if cat]


@router.get("", response_model=PartyListOut)
async def list_parties(
    category_name: Optional[str] = Query(None),
    service_id: Optional[uuid.UUID] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(12, ge=1, le=50),
    random: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    q = select(Party).options(
        selectinload(Party.host),
        selectinload(Party.members),
        selectinload(Party.service),
    )

    if service_id:
        q = q.where(Party.service_id == service_id)

    if category_name:
        q = q.join(Party.service).where(Service.category == category_name)

    if search:
        q = q.where(Party.title.ilike(f"%{search}%"))

    total = await db.scalar(select(func.count()).select_from(q.subquery())) or 0

    order = func.random() if random else Party.id.desc()
    q = q.offset((page - 1) * size).limit(size).order_by(order)

    result = await db.execute(q)
    parties = result.scalars().all()

    user_id = current_user.id if current_user else None
    return PartyListOut(
        parties=[_build_party_out(p, user_id) for p in parties],
        total=total,
        page=page,
        size=size,
    )


@router.get("/{party_id}", response_model=PartyOut)
async def get_party(
    party_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    result = await db.execute(
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.members),
            selectinload(Party.service),
        )
        .where(Party.id == party_id)
    )
    party = result.scalar_one_or_none()

    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    return _build_party_out(party, current_user.id if current_user else None)


@router.post("", response_model=PartyOut, status_code=status.HTTP_201_CREATED)
async def create_party(
    body: PartyCreate,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    svc = await db.get(Service, body.service_id)
    if not svc:
        raise HTTPException(status_code=404, detail="서비스를 찾을 수 없습니다.")

    if not svc.is_active:
        raise HTTPException(status_code=400, detail="비활성화된 서비스입니다.")

    max_members = body.max_members if body.max_members is not None else svc.max_members

    if max_members < 2:
        raise HTTPException(status_code=400, detail="최대 인원은 2명 이상이어야 합니다.")

    if max_members > svc.max_members:
        raise HTTPException(
            status_code=400,
            detail=f"최대 인원은 서비스 허용 인원({svc.max_members}명)을 초과할 수 없습니다.",
        )

    start_date = None
    end_date = None

    if body.start_date:
        try:
            start_date = date.fromisoformat(body.start_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="start_date 형식 오류 (YYYY-MM-DD)",
            )

    if body.end_date:
        try:
            end_date = date.fromisoformat(body.end_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="end_date 형식 오류 (YYYY-MM-DD)",
            )

    if start_date and end_date and end_date < start_date:
        raise HTTPException(
            status_code=400,
            detail="end_date는 start_date보다 빠를 수 없습니다.",
        )

    await consume_captcha_pass_token(body.captcha_pass_token)

    base_per_person = (svc.original_price or svc.monthly_price) / max_members
    commission = svc.commission_rate or 0.0
    monthly_per_person = round(base_per_person * (1 + commission))

    party = Party(
        leader_id=current_user.id,
        service_id=body.service_id,
        title=body.title,
        description=body.description,
        max_members=max_members,
        monthly_per_person=monthly_per_person,
        min_trust_score=body.min_trust_score if body.min_trust_score is not None else 0.0,
        status="recruiting",
        start_date=start_date,
        end_date=end_date,
    )

    try:
        db.add(party)
        await db.flush()

        await create_activity_log(
            db=db,
            user_id=current_user.id,
            action="CREATE_PARTY",
            description="파티 생성",
            metadata={"party_title": party.title},
            target_id=party.id,
        )

        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating party: {e}")
        raise HTTPException(
            status_code=500,
            detail="파티 생성 처리 중 서버 오류가 발생했습니다.",
        )

    result = await db.execute(
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.members),
            selectinload(Party.service),
        )
        .where(Party.id == party.id)
    )
    return _build_party_out(result.scalar_one(), current_user.id)


@router.post("/{party_id}/join", response_model=MessageOut)
async def apply_to_party(
    party_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Party).options(selectinload(Party.service)).where(Party.id == party_id)
    )
    party = result.scalar_one_or_none()

    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")

    if party.leader_id == current_user.id:
        raise HTTPException(status_code=400, detail="자신이 개설한 파티입니다.")

    existing_result = await db.execute(
        select(PartyMember).where(
            PartyMember.party_id == party_id,
            PartyMember.user_id == current_user.id,
        )
    )
    existing_row = existing_result.scalar_one_or_none()

    if existing_row is not None:
        current_status = (existing_row.status or "").lower()
        if current_status == "active":
            raise HTTPException(status_code=400, detail="이미 참여중인 파티입니다.")
        if current_status == "pending":
            raise HTTPException(status_code=400, detail="이미 신청하신 파티입니다. 리더 승인을 기다려주세요.")
        if current_status == "kicked":
            raise HTTPException(status_code=403, detail="이 파티에서 강퇴된 이력이 있어 재신청할 수 없습니다.")

    pending_count_row = await db.execute(
        select(func.count(PartyMember.id)).where(
            PartyMember.party_id == party_id,
            PartyMember.status == "active",
        )
    )
    active_count = pending_count_row.scalar() or 0
    effective_count = active_count + (1 if party.leader_id else 0)
    if effective_count >= (party.max_members or 0):
        raise HTTPException(status_code=400, detail="파티 인원이 가득 찼거나 대기자가 많습니다.")

    try:
        if existing_row is not None:
            existing_row.status = "pending"
            existing_row.leader_review_status = "pending"
            existing_row.join_type = "apply"
            existing_row.left_at = None
            existing_row.approved_at = None
            existing_row.rejected_at = None
        else:
            db.add(
                PartyMember(
                    party_id=party_id,
                    user_id=current_user.id,
                    role="member",
                    status="pending",
                    join_type="apply",
                    leader_review_status="pending",
                )
            )
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error applying to party: {e}")
        raise HTTPException(
            status_code=500,
            detail="파티 신청 처리 중 서버 오류가 발생했습니다.",
        )

    await notify_party_join_request_submitted(
        db=db,
        party=party,
        applicant_user_id=current_user.id,
    )

    await notify_party_join_requested(
        db=db,
        party=party,
        applicant_user_id=current_user.id,
        applicant_nickname=current_user.nickname,
    )

    return MessageOut(message="참여 신청이 완료되었습니다. 리더 승인을 기다려주세요.")


async def _load_party_with_members(db: AsyncSession, party_id: uuid.UUID) -> Party:
    result = await db.execute(
        select(Party)
        .options(
            selectinload(Party.host),
            selectinload(Party.service),
            selectinload(Party.members).selectinload(PartyMember.user),
        )
        .where(Party.id == party_id)
    )
    party = result.scalar_one_or_none()
    if not party:
        raise HTTPException(status_code=404, detail="파티를 찾을 수 없습니다.")
    return party


def _is_active_member(m: PartyMember) -> bool:
    return (m.status or "active").lower() == "active"


@router.get("/{party_id}/members", response_model=PartyMembersOut)
async def list_party_members(
    party_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = await _load_party_with_members(db, party_id)

    is_leader = (party.leader_id == current_user.id)
    is_member = any(_is_active_member(m) and m.user_id == current_user.id for m in party.members)
    if not (is_leader or is_member):
        raise HTTPException(status_code=403, detail="파티에 속해있지 않습니다.")

    out: list[PartyMemberOut] = []
    if party.host:
        out.append(PartyMemberOut(
            user_id=party.leader_id,
            nickname=party.host.nickname,
            role="leader",
            is_current_user=(party.leader_id == current_user.id),
        ))
    for m in party.members:
        if not _is_active_member(m):
            continue
        if m.user_id == party.leader_id:
            continue
        out.append(PartyMemberOut(
            user_id=m.user_id,
            nickname=m.user.nickname if m.user else None,
            role=m.role or "member",
            is_current_user=(m.user_id == current_user.id),
        ))
    return PartyMembersOut(members=out)


@router.post("/{party_id}/leave", response_model=MessageOut)
async def leave_party(
    party_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = await _load_party_with_members(db, party_id)

    if party.leader_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="리더는 먼저 리더 위임 후 탈퇴할 수 있습니다.",
        )

    member = next(
        (m for m in party.members if m.user_id == current_user.id and _is_active_member(m)),
        None,
    )
    if not member:
        raise HTTPException(status_code=404, detail="이 파티에 속해있지 않습니다.")

    try:
        member.status = "left"
        member.left_at = func.now()
        await _sync_member_count(db, party)
        party.current_members = max(0, party.current_members - 1)

        await create_activity_log(
            db=db,
            user_id=current_user.id,
            action="LEAVE_PARTY",
            description="파티 탈퇴",
            metadata={"party_title": party.title},
            target_id=party.id,
        )

        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error leaving party: {e}")
        raise HTTPException(status_code=500, detail="파티 탈퇴 처리 중 오류가 발생했습니다.")

    await _broadcast_party_updated(party_id)
    return MessageOut(message="파티에서 탈퇴했습니다.")


@router.delete("/{party_id}/members/{user_id}", response_model=MessageOut)
async def kick_member(
    party_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = await _load_party_with_members(db, party_id)

    if party.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="리더만 강퇴할 수 있습니다.")
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="자기 자신은 강퇴할 수 없습니다.")
    if user_id == party.leader_id:
        raise HTTPException(status_code=400, detail="리더는 강퇴할 수 없습니다.")

    target = next(
        (m for m in party.members if m.user_id == user_id and _is_active_member(m)),
        None,
    )
    if not target:
        raise HTTPException(status_code=404, detail="대상 멤버를 찾을 수 없습니다.")

    try:
        target.status = "kicked"
        target.left_at = func.now()
        await _sync_member_count(db, party)
        party.current_members = max(0, party.current_members - 1)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error kicking member: {e}")
        raise HTTPException(status_code=500, detail="강퇴 처리 중 오류가 발생했습니다.")

    await notify_party_member_kicked(
        db=db,
        party=party,
        target_user_id=user_id,
    )

    await _broadcast_party_updated(party_id)
    return MessageOut(message="멤버를 강퇴했습니다.")


@router.post("/{party_id}/transfer-leader", response_model=MessageOut)
async def transfer_leader(
    party_id: uuid.UUID,
    body: TransferLeaderRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = await _load_party_with_members(db, party_id)

    if party.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="리더만 위임할 수 있습니다.")
    if body.new_leader_user_id == current_user.id:
        raise HTTPException(status_code=400, detail="자기 자신에게 위임할 수 없습니다.")

    target = next(
        (m for m in party.members
         if m.user_id == body.new_leader_user_id and _is_active_member(m)),
        None,
    )
    if not target:
        raise HTTPException(status_code=404, detail="대상 멤버를 찾을 수 없습니다.")

    try:
        old_leader_row = next(
            (m for m in party.members if m.user_id == current_user.id),
            None,
        )
        if old_leader_row is None:
            db.add(PartyMember(
                party_id=party.id,
                user_id=current_user.id,
                role="member",
                status="active",
            ))
        else:
            old_leader_row.role = "member"
            old_leader_row.status = "active"
            old_leader_row.left_at = None

        target.role = "leader"
        party.leader_id = body.new_leader_user_id

        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error transferring leader: {e}")
        raise HTTPException(status_code=500, detail="리더 위임 처리 중 오류가 발생했습니다.")

    return MessageOut(message="리더를 위임했습니다.")


@router.get("/{party_id}/applications", response_model=PartyMembersOut)
async def list_applications(
    party_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = await _load_party_with_members(db, party_id)
    if party.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="리더만 조회할 수 있습니다.")

    items = [
        PartyMemberOut(
            user_id=m.user_id,
            nickname=(m.user.nickname if m.user else None),
            role=m.role or "member",
            is_current_user=False,
        )
        for m in party.members
        if (m.status or "").lower() == "pending"
    ]
    return PartyMembersOut(members=items)


@router.post("/{party_id}/applications/{user_id}/approve", response_model=MessageOut)
async def approve_application(
    party_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = await _load_party_with_members(db, party_id)
    if party.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="리더만 승인할 수 있습니다.")

    target = next(
        (m for m in party.members
         if m.user_id == user_id and (m.status or "").lower() == "pending"),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="대기 중인 신청을 찾을 수 없습니다.")

    current_count = await _sync_member_count(db, party)
    if current_count >= (party.max_members or 0):
        raise HTTPException(status_code=400, detail="파티 정원이 가득 차서 승인할 수 없습니다.")

    try:
        target.status = "active"
        target.leader_review_status = "approved"
        target.approved_at = func.now()
        target.rejected_at = None
        party.current_members = current_count + 1

        await create_activity_log(
            db=db,
            user_id=user_id,
            action="JOIN_PARTY",
            description="파티 참가",
            metadata={"party_title": party.title},
            target_id=party.id,
        )

        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error approving application: {e}")
        raise HTTPException(status_code=500, detail="승인 처리 중 오류가 발생했습니다.")

    await notify_party_join_approved(
        db=db,
        party=party,
        member_user_id=user_id,
    )

    await notify_party_member_joined_to_leader(
        db=db,
        party=party,
        member_user_id=user_id,
        member_nickname=target.user.nickname if target.user else None,
        join_type=target.join_type,
    )

    await _broadcast_party_updated(party_id)
    return MessageOut(message="신청을 승인했습니다.")


@router.post("/{party_id}/applications/{user_id}/reject", response_model=MessageOut)
async def reject_application(
    party_id: uuid.UUID,
    user_id: uuid.UUID,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    party = await _load_party_with_members(db, party_id)
    if party.leader_id != current_user.id:
        raise HTTPException(status_code=403, detail="리더만 거절할 수 있습니다.")

    target = next(
        (m for m in party.members
         if m.user_id == user_id and (m.status or "").lower() == "pending"),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="대기 중인 신청을 찾을 수 없습니다.")

    try:
        target.status = "rejected"
        target.leader_review_status = "rejected"
        target.rejected_at = func.now()
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.error(f"Error rejecting application: {e}")
        raise HTTPException(status_code=500, detail="거절 처리 중 오류가 발생했습니다.")

    await notify_party_join_request_rejected(
        db=db,
        party=party,
        applicant_user_id=user_id,
    )

    return MessageOut(message="신청을 거절했습니다.")
