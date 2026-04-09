import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from minio.error import S3Error

from core.minio_assets import fetch_minio_object

router = APIRouter()


@router.get("/assets/{bucket}/{object_name:path}")
async def get_asset(bucket: str, object_name: str):
    try:
        asset_bytes, content_type = await asyncio.to_thread(
            fetch_minio_object,
            bucket,
            object_name,
        )
    except S3Error as exc:
        if exc.code in {"NoSuchKey", "NoSuchBucket", "NoSuchObject"}:
            raise HTTPException(status_code=404, detail="Asset not found") from exc
        raise HTTPException(status_code=502, detail="Asset proxy failed") from exc

    return Response(
        content=asset_bytes,
        media_type=content_type,
        headers={"Cache-Control": "public, max-age=300"},
    )
