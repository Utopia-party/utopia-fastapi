from typing import Optional
from urllib.parse import urlparse

from fastapi import HTTPException, Request, status
from sqlalchemy import text

from core.database import AsyncSessionLocal
from schemas.saas import SaasKeyContext


# ── 유틸 ────────────────────────────────────────────────────────────────────

def _extract_host(url_or_host: str) -> str:
    try:
        parsed = urlparse(url_or_host)
        hostname = parsed.hostname
        return hostname if hostname else url_or_host.split("/")[0]
    except Exception:
        return url_or_host


def _match_domain(request_host: str, allowed_domains: list[str]) -> Optional[str]:
    if allowed_domains is None:
        return None
    request_host = request_host.lower().strip()
    for allowed in allowed_domains:
        allowed_lower = allowed.lower().strip()
        if request_host == allowed_lower:
            return allowed
        if allowed_lower.startswith("*."):
            suffix = allowed_lower[1:]
            if request_host.endswith(suffix):
                return allowed
        if allowed_lower == "localhost" and request_host.startswith("localhost"):
            return allowed
    return None


# ── DB 조회 ─────────────────────────────────────────────────────────────────

async def _lookup_by_api_key(api_key: str) -> Optional[dict]:
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("""
            SELECT id, client_name, api_key, secret_key, service_type,
                   allowed_domains, monthly_limit, current_month_usage, plan, is_active
            FROM saas_api_keys
            WHERE api_key = :api_key
            LIMIT 1
            """),
            {"api_key": api_key},
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def lookup_saas_secret(secret: str) -> Optional[dict]:
    """secret_key로 조회 — siteverify 용"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("""
            SELECT id, client_name, api_key, secret_key, service_type,
                   allowed_domains, monthly_limit, current_month_usage, plan, is_active
            FROM saas_api_keys
            WHERE secret_key = :secret
            LIMIT 1
            """),
            {"secret": secret},
        )
        row = result.mappings().first()
        return dict(row) if row else None


# ── 사용량 증가 + 로그 ────────────────────────────────────────────────────────

async def _increment_usage(api_key_id: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(
            text("""
            UPDATE saas_api_keys
            SET current_month_usage = current_month_usage + 1
            WHERE id = :id
            """),
            {"id": api_key_id},
        )
        await db.commit()


async def log_saas_usage(
    api_key_id: str,
    endpoint: str,
    client_ip: str,
    origin_domain: Optional[str],
    status_code: int,
    response_time_ms: int,
) -> None:
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                text("""
                INSERT INTO saas_api_usage_logs
                    (api_key_id, endpoint, client_ip, origin_domain,
                     status_code, response_time_ms)
                VALUES
                    (:api_key_id, :endpoint, CAST(:client_ip AS INET), :origin_domain,
                     :status_code, :response_time_ms)
                """),
                {
                    "api_key_id": api_key_id,
                    "endpoint": endpoint,
                    "client_ip": client_ip or "0.0.0.0",
                    "origin_domain": origin_domain,
                    "status_code": status_code,
                    "response_time_ms": response_time_ms,
                },
            )
            await db.commit()
    except Exception:
        pass


# ── 메인 검증 ────────────────────────────────────────────────────────────────

async def verify_saas_key(
    request: Request,
    required_service: Optional[str] = None,   # 'captcha_l2' | 'chat_filter' | None(둘 다 허용)
) -> Optional[SaasKeyContext]:
    """
    X-Saas-Key 헤더 검증.
    - 없으면 None (내부 호출 fallback)
    - required_service 지정 시 service_type 불일치이면 403
    """
    saas_key = request.headers.get("X-Saas-Key") or request.query_params.get("saas_key")

    if not saas_key:
        return None

    row = await _lookup_by_api_key(saas_key)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_codes": ["invalid-saas-key"]},
        )

    if not row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_codes": ["saas-key-disabled"]},
        )

    if required_service and row["service_type"] != required_service:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_codes": ["service-type-mismatch"]},
        )

    # 도메인 매칭
    origin = request.headers.get("Origin")
    referer = request.headers.get("Referer")
    origin_host = _extract_host(origin) if origin else None
    referer_host = _extract_host(referer) if referer else None
    request_host = origin_host or referer_host

    matched_domain = None
    if row["allowed_domains"] and request_host:
        matched_domain = _match_domain(request_host, row["allowed_domains"])
        if matched_domain is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error_codes": ["hostname-mismatch"]},
            )

    # 쿼터 체크
    if row["current_month_usage"] >= row["monthly_limit"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error_codes": ["quota-exceeded"]},
        )

    await _increment_usage(str(row["id"]))

    client_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or request.headers.get("X-Real-IP")
        or (request.client.host if request.client else "0.0.0.0")
    )
    await log_saas_usage(
        api_key_id=str(row["id"]),
        endpoint=request.url.path,
        client_ip=client_ip,
        origin_domain=request_host,
        status_code=200,
        response_time_ms=0,
    )

    return SaasKeyContext(
        api_key_id=str(row["id"]),
        client_name=row["client_name"],
        api_key=row["api_key"],
        secret_key=row["secret_key"],
        service_type=row["service_type"],
        plan=row["plan"],
        allowed_domains=row["allowed_domains"],
        monthly_limit=row["monthly_limit"],
        current_month_usage=row["current_month_usage"],
        matched_domain=matched_domain,
    )
