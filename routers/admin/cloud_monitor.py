import asyncio
import logging
import httpx
from fastapi import APIRouter, Depends, HTTPException

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
from core.config import settings
from routers.admin.deps import require_admin_context

router = APIRouter(prefix="/admin/cloud-monitor", tags=["admin-cloud-monitor"])

KAKAO_MONITOR_BASE = f"https://monitoring.{settings.KAKAO_CLOUD_REGION}.kakaocloud.com"
METRIC_EXPORT_URL = f"{KAKAO_MONITOR_BASE}/metric-export/grafana/{settings.KAKAO_CLOUD_PROJECT_ID}/prometheus/api/v1"


def _kc_headers(service_type: str = "server") -> dict:
    return {
        "Content-Type": "application/x-www-form-urlencoded",
        "Credential-ID": settings.KAKAO_CLOUD_CREDENTIAL_ID,
        "Credential-Secret": settings.KAKAO_CLOUD_CREDENTIAL_SECRET,
        "service-type": service_type,
    }


async def _query_metric(metric: str, service_type: str = "server") -> dict:
    """Prometheus instant query"""
    if not settings.KAKAO_CLOUD_PROJECT_ID or not settings.KAKAO_CLOUD_CREDENTIAL_ID:
        raise HTTPException(status_code=503, detail="카카오클라우드 자격증명이 설정되지 않았습니다.")
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{METRIC_EXPORT_URL}/query",
            headers=_kc_headers(service_type),
            data={"query": metric},
        )
        resp.raise_for_status()
        return resp.json()


async def _query_range(metric: str, start: str, end: str, step: str = "60", service_type: str = "server") -> dict:
    """Prometheus range query"""
    if not settings.KAKAO_CLOUD_PROJECT_ID or not settings.KAKAO_CLOUD_CREDENTIAL_ID:
        raise HTTPException(status_code=503, detail="카카오클라우드 자격증명이 설정되지 않았습니다.")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{METRIC_EXPORT_URL}/query_range",
            headers=_kc_headers(service_type),
            data={"query": metric, "start": start, "end": end, "step": step},
        )
        resp.raise_for_status()
        return resp.json()


def _extract_values(prom_result: dict) -> list:
    """Prometheus 응답에서 결과 리스트 추출"""
    try:
        return prom_result.get("data", {}).get("result", [])
    except Exception:
        return []


@router.get("/summary")
async def get_cloud_summary(_: object = Depends(require_admin_context)):
    """서버별 CPU / 메모리 / 네트워크 현재값 요약 (병렬 요청)"""
    METRIC_LABELS = [
        ("cpu_usage",               "cpu"),
        ("mem_usage",               "mem"),
        ("mem_used",                "mem_used"),
        ("mem_total",               "mem_total"),
        ("network_rx_bytes_persec", "net_in"),
        ("network_tx_bytes_persec", "net_out"),
        ("disk_used_percent",       "disk"),
        ("disk_used",               "disk_used"),
        ("disk_total",              "disk_total"),
    ]

    async def _fetch(metric_name: str, label: str):
        try:
            result = await _query_metric(metric_name)
            return label, _extract_values(result), None
        except Exception as e:
            return label, [], f"{metric_name}: {str(e)}"

    results = await asyncio.gather(*[_fetch(m, l) for m, l in METRIC_LABELS])

    metrics = {}
    errors = []
    for label, values, err in results:
        metrics[label] = values
        if err:
            errors.append(err)

    return {"metrics": metrics, "errors": errors}


@router.get("/range")
async def get_metric_range(
    metric: str,
    start: str,
    end: str,
    step: str = "60",
    service_type: str = "server",
    instance_id: str = "",
    _: object = Depends(require_admin_context),
):
    """
    특정 메트릭 시계열 데이터 조회
    - metric: cpu_usage / mem_usage / network_rx_bytes_persec 등
    - start/end: Unix timestamp (초)
    - step: 집계 간격(초)
    - instance_id: 특정 서버 UUID (없으면 전체 합산)
    """
    ALLOWED_METRICS = {
        "cpu_usage", "cpu_usage_user", "cpu_usage_system",
        "mem_usage", "mem_used",
        "network_rx_bytes_persec", "network_tx_bytes_persec",
        "disk_used_percent", "disk_read_bytes_persec", "disk_write_bytes_persec",
    }
    if metric not in ALLOWED_METRICS:
        raise HTTPException(status_code=400, detail=f"허용되지 않는 메트릭입니다: {metric}")

    # 서버 필터: instance_id가 있으면 PromQL에 레이블 필터 추가
    if instance_id:
        query = f'{metric}{{instance_id="{instance_id}"}}'
    else:
        query = metric

    try:
        result = await _query_range(query, start, end, step, service_type)
        return {"result": _extract_values(result)}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"카카오클라우드 API 오류: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/lb")
async def get_lb_metrics(_: object = Depends(require_admin_context)):
    """로드밸런서 트래픽 현재값"""
    metrics = {}
    for metric_name, label in [
        ("lb_bytes_in_persec", "bytes_in"),
        ("lb_bytes_out_persec", "bytes_out"),
        ("lb_active_connections", "active_conn"),
        ("lb_new_connections_persec", "new_conn"),
    ]:
        try:
            result = await _query_metric(metric_name, service_type="lb")
            metrics[label] = _extract_values(result)
        except Exception as e:
            metrics[label] = []

    return {"metrics": metrics}


@router.get("/debug/labels")
async def debug_metric_labels(_: object = Depends(require_admin_context)):
    """실제 메트릭 레이블 키 확인용 (인스턴스명 문제 디버깅)"""
    result = await _query_metric("cpu_usage")
    raw = result.get("data", {}).get("result", [])
    labels = [r.get("metric", {}) for r in raw[:5]]
    return {"labels": labels, "count": len(raw)}


@router.get("/debug/raw")
async def debug_raw_values(_: object = Depends(require_admin_context)):
    """disk_used, disk_total, mem_used, mem_total 실제 값 확인"""
    result = {}
    for m in ["disk_used", "disk_total", "mem_used", "mem_total"]:
        try:
            r = await _query_metric(m)
            vals = _extract_values(r)
            result[m] = [{"metric": v.get("metric", {}), "value": v.get("value", [])} for v in vals[:2]]
        except Exception as e:
            result[m] = str(e)
    return result
