from urllib.parse import quote

from core.config import settings

DEFAULT_SERVICE_LOGO_BUCKET = "service-logos"


def build_minio_asset_url(
    asset_key: str | None,
    *,
    default_bucket: str = DEFAULT_SERVICE_LOGO_BUCKET,
) -> str | None:
    if not asset_key:
        return None

    normalized = asset_key.strip().lstrip("/")
    if not normalized:
        return None

    if "/" in normalized:
        bucket, object_name = normalized.split("/", 1)
    else:
        bucket, object_name = default_bucket, normalized

    endpoint = (settings.MINIO_PUBLIC_ENDPOINT or settings.MINIO_ENDPOINT).strip().rstrip("/")
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        base_url = endpoint
    else:
        protocol = "https" if settings.MINIO_SECURE else "http"
        base_url = f"{protocol}://{endpoint}"

    return f"{base_url}/{bucket}/{quote(object_name, safe='/')}"
