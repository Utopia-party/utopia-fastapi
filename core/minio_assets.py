import mimetypes
from urllib.parse import quote

from minio import Minio
from core.config import settings

DEFAULT_SERVICE_LOGO_BUCKET = "service-logos"


def split_minio_asset_key(
    asset_key: str | None,
    *,
    default_bucket: str = DEFAULT_SERVICE_LOGO_BUCKET,
) -> tuple[str, str] | None:
    if not asset_key:
        return None

    normalized = asset_key.strip().lstrip("/")
    if not normalized:
        return None

    if "/" in normalized:
        return normalized.split("/", 1)

    return default_bucket, normalized


def build_minio_asset_url(
    asset_key: str | None,
    *,
    default_bucket: str = DEFAULT_SERVICE_LOGO_BUCKET,
) -> str | None:
    parsed = split_minio_asset_key(asset_key, default_bucket=default_bucket)
    if not parsed:
        return None

    bucket, object_name = parsed
    return f"/api/assets/{bucket}/{quote(object_name, safe='/')}"


def build_minio_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def fetch_minio_object(bucket: str, object_name: str) -> tuple[bytes, str]:
    client = build_minio_client()
    response = client.get_object(bucket, object_name)

    try:
        image_bytes = response.read()
        content_type = (
            response.headers.get("Content-Type")
            or mimetypes.guess_type(object_name)[0]
            or "application/octet-stream"
        )
        return image_bytes, content_type
    finally:
        response.close()
        response.release_conn()
