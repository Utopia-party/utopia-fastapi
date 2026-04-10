import io
import os
import uuid
from datetime import timedelta
from typing import Optional

from fastapi import HTTPException, UploadFile
from minio import Minio
from minio.error import S3Error
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.user import User
from schemas.mypage.profile import MyPageProfileResponse, UpdateMyPageProfileResponse

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
MAX_IMAGE_SIZE = 5 * 1024 * 1024


def _get_minio_client() -> Minio:
    """내부 전용 클라이언트 - 업로드/삭제용"""
    return Minio(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def _get_public_minio_client() -> Minio:
    """브라우저용 Presigned URL 생성 전용 클라이언트 (공인 IP로 서명)"""
    endpoint = settings.MINIO_PUBLIC_ENDPOINT or settings.MINIO_ENDPOINT
    # MinIO는 path 포함 endpoint 불가 → host만 추출
    endpoint = endpoint.split("/")[0]
    return Minio(
        endpoint=endpoint,
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


def _build_profile_image_url(profile_image_key: Optional[str]) -> Optional[str]:
    """공인 IP 클라이언트로 서명 후 path prefix 삽입"""
    if not profile_image_key:
        return None

    client = _get_public_minio_client()

    url = client.presigned_get_object(
        settings.PROFILE_MINIO_BUCKET,
        profile_image_key,
        expires=timedelta(hours=1),
    )

    # MINIO_PUBLIC_ENDPOINT에 path가 있으면 (예: 210.109.15.10/minio)
    # 서명된 URL에 해당 path prefix 삽입
    if settings.MINIO_PUBLIC_ENDPOINT and "/" in settings.MINIO_PUBLIC_ENDPOINT:
        public_host = settings.MINIO_PUBLIC_ENDPOINT.split("/")[0]
        path_prefix = "/" + "/".join(settings.MINIO_PUBLIC_ENDPOINT.split("/")[1:])
        # http://210.109.15.10/profile-images/... → http://210.109.15.10/minio/profile-images/...
        url = url.replace(f"{public_host}/", f"{public_host}{path_prefix}/", 1)

    return url


def _to_profile_response(user: User) -> MyPageProfileResponse:
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
    )


async def get_my_profile_service(
    db: AsyncSession,
    current_user: User,
) -> MyPageProfileResponse:
    return _to_profile_response(current_user)


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

    # 업로드/삭제는 내부 클라이언트 사용
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

    return UpdateMyPageProfileResponse(
        message="프로필이 수정되었습니다.",
        user=_to_profile_response(current_user),
    )
