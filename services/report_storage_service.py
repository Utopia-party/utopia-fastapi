from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile


REPORT_BUCKET = "report-evidence"


async def upload_report_file(file: UploadFile, report_id: str) -> dict:
    """
    실제 프로젝트의 MinIO 유틸로 교체하세요.
    반환 예시:
    {
        "object_key": "...",
        "original_filename": "...",
        "content_type": "...",
        "file_size": 1234
    }
    """
    ext = Path(file.filename or "").suffix
    object_key = f"reports/{report_id}/{uuid.uuid4()}{ext}"

    content = await file.read()
    file_size = len(content)

    # TODO: 실제 MinIO 업로드 코드로 교체
    # minio_client.put_object(...)

    return {
        "object_key": object_key,
        "original_filename": file.filename,
        "content_type": file.content_type,
        "file_size": file_size,
    }