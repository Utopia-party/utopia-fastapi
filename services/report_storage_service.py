from __future__ import annotations

import uuid
from datetime import timedelta
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile
from minio.error import S3Error

from core.config import settings
from core.minio import minio_client


REPORT_BUCKET = settings.MINIO_REPORT_BUCKET

MAX_REPORT_FILE_SIZE = 5 * 1024 * 1024  # 5MB

ALLOWED_REPORT_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "application/pdf",
}


def ensure_report_bucket_exists() -> None:
    """
    신고 증빙 파일용 MinIO bucket이 없으면 생성합니다.
    """
    try:
        if not minio_client.bucket_exists(REPORT_BUCKET):
            minio_client.make_bucket(REPORT_BUCKET)
    except S3Error as e:
        raise HTTPException(
            status_code=500,
            detail="신고 증빙 저장소 초기화에 실패했습니다.",
        ) from e


async def upload_report_file(file: UploadFile, report_id: str) -> dict:
    """
    신고 증빙 파일을 MinIO에 업로드하고,
    DB에 저장할 메타데이터를 반환합니다.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="파일명이 없는 파일은 업로드할 수 없습니다.",
        )

    content_type = file.content_type or "application/octet-stream"

    if content_type not in ALLOWED_REPORT_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="이미지 또는 PDF 파일만 업로드할 수 있습니다.",
        )

    ext = Path(file.filename).suffix.lower()
    object_key = f"reports/{report_id}/{uuid.uuid4()}{ext}"

    content = await file.read()
    file_size = len(content)

    if file_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="빈 파일은 업로드할 수 없습니다.",
        )

    if file_size > MAX_REPORT_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="파일은 5MB 이하만 업로드할 수 있습니다.",
        )

    try:
        ensure_report_bucket_exists()

        minio_client.put_object(
            bucket_name=REPORT_BUCKET,
            object_name=object_key,
            data=BytesIO(content),
            length=file_size,
            content_type=content_type,
        )

    except S3Error as e:
        raise HTTPException(
            status_code=500,
            detail="증빙 파일 업로드에 실패했습니다.",
        ) from e

    return {
        "object_key": object_key,
        "original_filename": file.filename,
        "content_type": content_type,
        "file_size": file_size,
    }


def get_report_file_presigned_url(
    object_key: str,
    expires_seconds: int = 300,
) -> str:
    """
    관리자 페이지에서 증빙 파일을 확인할 수 있도록
    짧은 만료 시간을 가진 조회 URL을 발급합니다.
    """
    try:
        ensure_report_bucket_exists()

        return minio_client.presigned_get_object(
            bucket_name=REPORT_BUCKET,
            object_name=object_key,
            expires=timedelta(seconds=expires_seconds),
        )

    except S3Error as e:
        raise HTTPException(
            status_code=500,
            detail="증빙 파일 조회 URL 생성에 실패했습니다.",
        ) from e