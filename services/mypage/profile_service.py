import io
import os
import uuid
from datetime import timedelta
from typing import Optional,Any

from fastapi import HTTPException, UploadFile
from minio import Minio
from minio.error import S3Error
from sqlalchemy import select, func, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.user import User
from models.party import PartyMember
from models.admin import ActivityLog
from schemas.mypage.profile import (
    MyPageProfileResponse,
    UpdateMyPageProfileResponse,
    RecentActivityItem,
)

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
MAX_IMAGE_SIZE = 5 * 1024 * 1024


def _get_minio_client() -> Minio:
    """내부 전용 클라이언트 - 업로드/삭제/Presigned URL 생성용"""
    return Minio(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def _ensure_bucket_exists(client: Minio, bucket_name: str) -> None:
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)


def _get_extension(filename: Optional[str], content_type: Optional[str]) -> str:
    extension = os.path.splitext(filename or "")[1].lower()
    if extension in {".jpg", ".jpeg", ".png", ".webp"}:
        return extension

    mapping = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return mapping.get(content_type or "", ".jpg")

# 프로필 이미지 
def _build_profile_image_url(profile_image_key: Optional[str]) -> Optional[str]:
    """
    내부 클라이언트로 Presigned URL 생성 후
    nginx proxy_set_header Host가 내부 IP로 바꿔주므로
    단순히 host 문자열만 외부 주소로 교체
    """
    if not profile_image_key:
        return None

    client = _get_minio_client()

    url = client.presigned_get_object(
        settings.PROFILE_MINIO_BUCKET,
        profile_image_key,
        expires=timedelta(hours=1),
    )

    # http://10.10.0.10:9000/... → http://210.109.15.10/minio/...
    if settings.MINIO_PUBLIC_ENDPOINT:
        url = url.replace(
            settings.MINIO_ENDPOINT,        # 10.10.0.10:9000
            settings.MINIO_PUBLIC_ENDPOINT, # 210.109.15.10/minio
        )

    return url

def _normalize_metadata(metadata: Any) -> dict:
    if isinstance(metadata, dict):
        return metadata
    return {}


def _to_recent_activity_item(activity: ActivityLog) -> RecentActivityItem:
    return RecentActivityItem(
        id=str(activity.id),
        action=activity.action_type,  
        description=activity.description,
        ip_address=str(activity.ip_address) if activity.ip_address else None,
        user_agent=activity.user_agent,
        metadata=_normalize_metadata(activity.extra_metadata),
        target_id=str(activity.target_id) if activity.target_id else None,
        created_at=activity.created_at,
    )

def _to_profile_response(
    user: User,
    total_party_participations: int = 0,
    active_party_count: int = 0,
    recommendation_count: int = 0,
    recent_activities: list[RecentActivityItem] | None = None,
) -> MyPageProfileResponse:
    return MyPageProfileResponse(
        user_id=str(user.id),
        email=user.email,
        name=user.name,
        nickname=user.nickname,
        phone=user.phone,
        provider=user.provider,
        role=user.role,
        trust_score=float(user.trust_score),
        profile_image=_build_profile_image_url(user.profile_image_key),
        created_at=user.created_at,
        total_party_participations=total_party_participations,
        active_party_count=active_party_count,
        recommendation_count=recommendation_count,
        recent_activities=recent_activities or [],
    )

async def _get_total_party_participations(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> int:
    stmt = (
        select(func.count(distinct(PartyMember.party_id)))
        .where(PartyMember.user_id == user_id)
    )
    result = await db.execute(stmt)
    return result.scalar() or 0


async def _get_active_party_count(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> int:
    stmt = (
        select(func.count(distinct(PartyMember.party_id)))
        .where(
            PartyMember.user_id == user_id,
            PartyMember.status == "active",
        )
    )
    result = await db.execute(stmt)
    return result.scalar() or 0

async def _get_recommendation_count(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> int:
    stmt = (
        select(func.count(User.id))
        .where(User.referrer_id == user_id)
    )
    result = await db.execute(stmt)
    return result.scalar() or 0

async def _get_recent_activities(
    db: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 5,
) -> list[RecentActivityItem]:
    stmt = (
        select(ActivityLog)
        .where(ActivityLog.actor_user_id == user_id)
        .order_by(ActivityLog.created_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    activities = result.scalars().all()
    return [_to_recent_activity_item(activity) for activity in activities]

async def get_my_profile_service(
    db: AsyncSession,
    current_user: User,
) -> MyPageProfileResponse:
    total_party_participations = await _get_total_party_participations(
        db=db,
        user_id=current_user.id,
    )
    active_party_count = await _get_active_party_count(
        db=db,
        user_id=current_user.id,
    )
    recommendation_count = await _get_recommendation_count(
        db=db,
        user_id=current_user.id,
    )
    recent_activities = await _get_recent_activities(
        db=db,
        user_id=current_user.id,
        limit=5,
    )

    return _to_profile_response(
        user=current_user,
        total_party_participations=total_party_participations,
        active_party_count=active_party_count,
        recommendation_count=recommendation_count,
        recent_activities=recent_activities,
    )

# 프로필 수정
async def update_my_profile_service(
    db: AsyncSession,
    current_user: User,
    nickname: str,
    phone: str,
    profile_image: Optional[UploadFile] = None,
    remove_profile_image: bool = False,
) -> UpdateMyPageProfileResponse:
    nickname = nickname.strip()
    phone = phone.strip()

    if not nickname:
        raise HTTPException(status_code=400, detail="닉네임은 필수입니다.")
    if not phone:
        raise HTTPException(status_code=400, detail="전화번호는 필수입니다.")

    if nickname != current_user.nickname:
        stmt = select(User).where(User.nickname == nickname, User.id != current_user.id)
        result = await db.execute(stmt)
        duplicate_user = result.scalar_one_or_none()
        if duplicate_user:
            raise HTTPException(status_code=409, detail="이미 사용 중인 닉네임입니다.")

    current_user.nickname = nickname
    current_user.phone = phone

    client = _get_minio_client()
    bucket_name = settings.PROFILE_MINIO_BUCKET
    _ensure_bucket_exists(client, bucket_name)

    if remove_profile_image and current_user.profile_image_key:
        try:
            client.remove_object(bucket_name, current_user.profile_image_key)
        except S3Error:
            pass
        current_user.profile_image_key = None

    if profile_image is not None and profile_image.filename:
        if profile_image.content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(status_code=400, detail="JPG, PNG, WEBP 형식만 업로드 가능합니다.")

        file_bytes = await profile_image.read()
        if len(file_bytes) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="프로필 이미지는 5MB 이하만 가능합니다.")

        extension = _get_extension(profile_image.filename, profile_image.content_type)
        object_name = f"profile/{current_user.id}_{uuid.uuid4().hex}{extension}"

        if current_user.profile_image_key:
            try:
                client.remove_object(bucket_name, current_user.profile_image_key)
            except S3Error:
                pass

        client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=profile_image.content_type,
        )

        current_user.profile_image_key = object_name

    await db.commit()
    await db.refresh(current_user)

    total_party_participations = await _get_total_party_participations(
        db=db,
        user_id=current_user.id,
    )
    active_party_count = await _get_active_party_count(
        db=db,
        user_id=current_user.id,
    )
    recommendation_count = await _get_recommendation_count(
        db=db,
        user_id=current_user.id,
    )
    recent_activities = await _get_recent_activities(
        db=db,
        user_id=current_user.id,
        limit=5,
    )

    return UpdateMyPageProfileResponse(
        message="프로필이 수정되었습니다.",
        user=_to_profile_response(
            user=current_user,
            total_party_participations=total_party_participations,
            active_party_count=active_party_count,
            recommendation_count=recommendation_count,
            recent_activities=recent_activities,
        ),
    )