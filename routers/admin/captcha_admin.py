from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from core.config import settings
from core.database import get_db
from core.redis_client import redis_client
from core.minio_assets import build_minio_asset_url
from core.security import require_user
from models.admin import (
    ActivityLog,
    AdminRole,
    ModerationAction,
    Receipt,
    Settlement,
    SystemLog,
)
from models.report import Report

from models.notification import Notification
from models.party import Party, PartyChat, PartyMember, Service
from models.payment import Payment
from models.quick_match.request import QuickMatchRequest
from models.refresh_token import RefreshToken
from models.mypage.trust_score import TrustScore
from models.user import User
from schemas.admin import (
    AdminDashboardOut,
    AdminModerationHistoryOut,
    AdminPartyActionIn,
    AdminPartyMemberKickIn,
    AdminPartyMemberOut,
    AdminPartyMemberRoleIn,
    AdminPartyRecordOut,
    AdminPermissionOut,
    ChatModerationLogOut,
    ChatModerationStatsOut,
    DashboardChartOut,
    DashboardRecentActivityOut,
    AdminRoleRecordOut,
    AdminRoleUpdateIn,
    AdminServiceRecordOut,
    AdminServiceUpdateIn,
    AdminStatusUpdateIn,
    AdminReportStatusUpdateIn,
    AdminUserAccessLogOut,
    AdminUserDetailOut,
    AdminUserRecordOut,
    AdminUserStatusLogOut,
    AdminUserTrustHistoryOut,
    AdminUserTrustScoreUpdateIn,
    AdminUserStatusUpdateIn,
    DashboardSeriesPointOut,
    ReceiptRecordOut,
    ReportRecordOut,
    SettlementRecordOut,
    SystemLogRecordOut,
    UserStatusLogOut,
)
from services.notifications.report_notification_service import (
    notify_report_result_to_reporter,
    notify_report_warning_to_target,
    notify_report_penalty_to_target,
)

from .deps import (
    AdminContext,
    require_admin_context,
    require_admin_user_permission,
    require_admin_party_permission,
    require_admin_report_permission,
    require_admin_receipt_permission,
    require_admin_settlement_permission,
    require_admin_payment_permission,
    require_admin_handocr_permission,
    require_admin_log_permission,
    require_admin_moderation_permission,
    require_admin_role_permission,
    _format_datetime, _format_relative, _to_int,
    _date_range_bounds, _format_change, _bucket_labels,
    _shift_comparison_range, _series_label,
    _user_display_name, _actor_display_name,
    _build_trust_history_detail, _moderation_action_label,
    _admin_permissions_for_role, _manual_status_label,
    _user_status_label, _party_status_label,
    _report_status_label, _report_status_code,
    _report_type_label, _report_target_counts_subquery,
    _receipt_status_label, _receipt_status_code,
    _settlement_status_label, _settlement_status_code,
    _append_activity_log, _append_system_log,
    _admin_permissions_payload, _has_any_admin_permission,
    _serialize_admin_permissions, _serialize_admin_role,
    _serialize_admin_service, _report_target_display_map,
    _assert_admin_permission, _latest_user_status_actions_subquery,
    _count_root_admins, _ensure_admin_role,
)

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/captcha/shadow", tags=["admin-captcha"])
async def get_shadow_mode(current_user: User = Depends(require_user)):
    """현재 LSTM Shadow Mode 상태 조회"""
    return {
        "shadow_mode": settings.LSTM_SHADOW_MODE,
        "lstm_weight": settings.LSTM_WEIGHT,
        "score_formula": (
            "rule × {r:.0%} + KNN × {k:.0%} + LSTM × {l:.0%}".format(
                r=1.0 - settings.LSTM_WEIGHT - 0.2,
                k=0.2,
                l=settings.LSTM_WEIGHT,
            )
            if not settings.LSTM_SHADOW_MODE
            else "rule × (1-knn_w) + KNN × knn_w  (LSTM 로그만)"
        ),
    }


@router.put("/captcha/shadow", tags=["admin-captcha"])
async def toggle_shadow_mode(current_user: User = Depends(require_user)):
    """LSTM Shadow Mode ON/OFF 토글 (런타임 변경)"""
    settings.LSTM_SHADOW_MODE = not settings.LSTM_SHADOW_MODE
    new_state = settings.LSTM_SHADOW_MODE

    return {
        "shadow_mode": new_state,
        "message": (
            "LSTM Shadow ON — LSTM은 로그만 기록, final_score에 미반영"
            if new_state
            else "LSTM Shadow OFF — LSTM이 final_score에 반영됨 "
                 f"(rule×{1.0 - settings.LSTM_WEIGHT - 0.2:.0%} + KNN×20% + LSTM×{settings.LSTM_WEIGHT:.0%})"
        ),
    }


# ── IP 제재 관리 ──────────────────────────────────────

_CAPTCHA_KEY_PREFIXES = [
    "captcha:lock:",
    "captcha:lock-count:",
    "captcha:ban:",
    "captcha:wait:",
    "captcha:force-challenge:",
]


@router.get("/captcha/blocked-ips", tags=["admin-captcha"])
async def list_blocked_ips(current_user: User = Depends(require_user)):
    """현재 잠금/밴 상태인 IP 목록 조회"""
    blocked: dict[str, dict] = {}

    for prefix in _CAPTCHA_KEY_PREFIXES:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=f"{prefix}*", count=100)
            for key in keys:
                key_str = key if isinstance(key, str) else key.decode()
                ip = key_str.replace(prefix, "")
                if ip not in blocked:
                    blocked[ip] = {"ip": ip, "lock": False, "ban": False, "wait": False, "lock_count": 0, "ttl": {}}

                ttl = await redis_client.ttl(key_str)

                if prefix == "captcha:lock:":
                    blocked[ip]["lock"] = True
                    blocked[ip]["ttl"]["lock"] = ttl
                elif prefix == "captcha:ban:":
                    blocked[ip]["ban"] = True
                    blocked[ip]["ttl"]["ban"] = ttl
                elif prefix == "captcha:wait:":
                    blocked[ip]["wait"] = True
                    blocked[ip]["ttl"]["wait"] = ttl
                elif prefix == "captcha:lock-count:":
                    val = await redis_client.get(key_str)
                    blocked[ip]["lock_count"] = int(val) if val else 0

            if cursor == 0:
                break

    # ban > lock > wait 우선순위로 정렬
    items = sorted(
        blocked.values(),
        key=lambda x: (x["ban"], x["lock"], x["wait"]),
        reverse=True,
    )
    return {"blocked_ips": items, "total": len(items)}


@router.delete("/captcha/blocked-ips/{ip}", tags=["admin-captcha"])
async def unblock_ip(ip: str, current_user: User = Depends(require_user)):
    """특정 IP의 모든 캡챠 제재 해제"""
    deleted_keys = []
    for prefix in _CAPTCHA_KEY_PREFIXES:
        key = f"{prefix}{ip}"
        result = await redis_client.delete(key)
        if result:
            deleted_keys.append(key)

    return {
        "ip": ip,
        "unblocked": len(deleted_keys) > 0,
        "deleted_keys": deleted_keys,
        "message": f"{ip} 제재 해제 완료" if deleted_keys else f"{ip}에 대한 제재가 없습니다",
    }


@router.delete("/captcha/blocked-ips", tags=["admin-captcha"])
async def unblock_all_ips(current_user: User = Depends(require_user)):
    """모든 IP의 캡챠 제재 해제 (FLUSHDB 대신 캡챠 키만 삭제)"""
    total_deleted = 0
    for prefix in _CAPTCHA_KEY_PREFIXES:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=f"{prefix}*", count=100)
            if keys:
                await redis_client.delete(*keys)
                total_deleted += len(keys)
            if cursor == 0:
                break

    return {
        "total_deleted": total_deleted,
        "message": f"캡챠 제재 {total_deleted}건 전체 해제 완료",
    }


# ── 캡챠 수치 설정 (런타임) ──────────────────────────────

@router.get("/captcha/config", tags=["admin-captcha"])
async def get_captcha_config(current_user: User = Depends(require_user)):
    """현재 캡챠 가중치 및 임계값 조회"""
    lstm_w = getattr(settings, "LSTM_WEIGHT", 0.7)
    knn_w = getattr(settings, "KNN_WEIGHT", 0.2)
    rule_w = round(1.0 - lstm_w - knn_w, 4)
    return {
        "lstm_weight": lstm_w,
        "knn_weight": knn_w,
        "rule_weight": rule_w,
        "pass_threshold": getattr(settings, "CAPTCHA_PASS_THRESHOLD", 0.7),
        "challenge_threshold": getattr(settings, "CAPTCHA_CHALLENGE_THRESHOLD", 0.3),
    }


@router.put("/captcha/config", tags=["admin-captcha"])
async def update_captcha_config(
    body: dict,
    current_user: User = Depends(require_user),
):
    """캡챠 가중치/임계값 런타임 변경

    body 예시:
      {"lstm_weight": 0.7, "knn_weight": 0.2,
       "pass_threshold": 0.7, "challenge_threshold": 0.3}
    """
    import services.captcha_service as cs

    updated: list[str] = []

    # ── 가중치 변경 ──
    if "lstm_weight" in body or "knn_weight" in body:
        lstm_w = float(body.get("lstm_weight", settings.LSTM_WEIGHT))
        knn_w = float(body.get("knn_weight", getattr(settings, "KNN_WEIGHT", 0.2)))
        rule_w = 1.0 - lstm_w - knn_w

        if rule_w < 0 or rule_w > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"rule_weight({rule_w:.2f})가 0~1 범위를 벗어납니다. lstm+knn <= 1.0 이어야 합니다.",
            )

        settings.LSTM_WEIGHT = lstm_w
        settings.KNN_WEIGHT = knn_w
        updated.append(f"weights: rule={rule_w:.0%} KNN={knn_w:.0%} LSTM={lstm_w:.0%}")

    # ── 임계값 변경 ──
    if "pass_threshold" in body:
        val = float(body["pass_threshold"])
        settings.CAPTCHA_PASS_THRESHOLD = val
        cs.CAPTCHA_PASS_THRESHOLD = val
        updated.append(f"pass_threshold={val}")

    if "challenge_threshold" in body:
        val = float(body["challenge_threshold"])
        settings.CAPTCHA_CHALLENGE_THRESHOLD = val
        cs.CAPTCHA_CHALLENGE_THRESHOLD = val
        updated.append(f"challenge_threshold={val}")

    lstm_w = settings.LSTM_WEIGHT
    knn_w = getattr(settings, "KNN_WEIGHT", 0.2)
    rule_w = round(1.0 - lstm_w - knn_w, 4)

    return {
        "message": f"변경 완료: {', '.join(updated)}" if updated else "변경 사항 없음",
        "lstm_weight": lstm_w,
        "knn_weight": knn_w,
        "rule_weight": rule_w,
        "pass_threshold": settings.CAPTCHA_PASS_THRESHOLD,
        "challenge_threshold": settings.CAPTCHA_CHALLENGE_THRESHOLD,
    }


# ── 챌린지 강제 발동 ──────────────────────────────────────

@router.post("/captcha/force-challenge", tags=["admin-captcha"])
async def force_challenge(
    body: dict | None = None,
    current_user: User = Depends(require_user),
):
    """특정 IP(또는 모든 IP)에 대해 다음 캡챠를 강제로 challenge로 만듦.

    body 예시:
      {"ip": "202.31.255.26"}   → 해당 IP만
      {} 또는 body 없음          → 모든 활성 IP
    """
    target_ip = (body or {}).get("ip", "").strip()

    if target_ip:
        # 특정 IP만
        await redis_client.setex(f"captcha:force-challenge:{target_ip}", 300, "1")
        return {"message": f"{target_ip}에 챌린지 강제 발동 (5분간 유효)"}

    # IP 미지정 시: 와일드카드 대신 잘 알려진 방법 → 0.0.0.0 플래그
    await redis_client.setex("captcha:force-challenge:*", 300, "1")
    return {"message": "모든 IP에 챌린지 강제 발동 (5분간 유효)"}

# ─── 결제 내역 관리 ───────────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel

class AdminPaymentRecordOut(_BaseModel):
    id: str
    userId: str
    userNickname: str
    userName: str | None
    partyId: str
    partyTitle: str
    serviceName: str | None
    role: str
    basePrice: int
    amount: int
    discountReason: str | None
    commissionRate: float
    commissionAmount: int
    paymentMethod: str | None
    status: str
    billingMonth: str
    pricingType: str | None
    paidAt: str | None
    createdAt: str

    class Config:
        from_attributes = True


class AdminPaymentListOut(_BaseModel):
    items: list[AdminPaymentRecordOut]
    total: int
    page: int
    limit: int
    totalPages: int


def _admin_payment_total_price(
    payment: Payment,
    party: Party,
    service: Service | None,
) -> int:
    if service and service.monthly_price:
        return int(service.monthly_price)
    if payment.base_price:
        return int(payment.base_price)
    if party.monthly_per_person and party.max_members:
        return int(party.monthly_per_person * party.max_members)
    return int(payment.amount)


def _admin_payment_per_person_price(
    payment: Payment,
    party: Party,
    service: Service | None,
) -> int:
    total_price = _admin_payment_total_price(payment, party, service)
    max_members = int(party.max_members or 0)
    if max_members > 0:
        return max(1, round(total_price / max_members))
    if party.monthly_per_person:
        return int(party.monthly_per_person)
    return int(payment.amount)


def _admin_payment_display_amount(
    payment: Payment,
    user: User,
    party: Party,
    service: Service | None,
) -> tuple[int, int]:
    per_person_price = _admin_payment_per_person_price(payment, party, service)
    discount_rate = 0.0

    if party.leader_id == user.id and service and service.leader_discount_rate:
        discount_rate += float(service.leader_discount_rate or 0.0)

    if user.referrer_id and service and service.referral_discount_rate:
        discount_rate += float(service.referral_discount_rate or 0.0)

    discount_rate = min(discount_rate, 1.0)
    actual_amount = round(per_person_price * (1 - discount_rate))
    return per_person_price, actual_amount


@router.get("/captcha/stats", tags=["admin-captcha"])
async def get_captcha_stats(
    period: str = Query(default="daily", pattern="^(daily|weekly|monthly)$"),
    start_date: str | None = Query(default=None, description="시작일 (YYYY-MM-DD)"),
    end_date: str | None = Query(default=None, description="종료일 (YYYY-MM-DD)"),
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """캡챠 대시보드 통계: 기간별 요약, 점수 분포, 추이

    - period: daily(일별) / weekly(주별) / monthly(월별)
    - start_date, end_date: 조회 기간 (미지정 시 기본값 적용)
      - daily: 최근 7일
      - weekly: 최근 8주
      - monthly: 최근 6개월
    - summary: 선택 기간 전체의 pass/challenge/block 건수 및 비율
    - score_distribution: 선택 기간 내 점수 히스토그램
    - trend: 기간 단위별 추이 데이터
    """
    from datetime import date as date_type

    # ── 기간 계산 ──
    if start_date and end_date:
        try:
            s_date = date_type.fromisoformat(start_date)
            e_date = date_type.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="날짜 형식이 올바르지 않습니다 (YYYY-MM-DD)")
    else:
        e_date = date.today()
        if period == "daily":
            s_date = e_date - timedelta(days=6)
        elif period == "weekly":
            s_date = e_date - timedelta(weeks=8)
        else:  # monthly
            s_date = e_date - timedelta(days=180)

    # asyncpg는 문자열→date CAST 불가 → Python date 객체를 직접 바인딩
    # end는 해당일 23:59:59까지 포함하기 위해 +1일의 datetime으로 변환
    from datetime import datetime as dt_type

    start_dt = dt_type(s_date.year, s_date.month, s_date.day)
    end_dt = dt_type(e_date.year, e_date.month, e_date.day) + timedelta(days=1)
    params = {"start": start_dt, "end": end_dt}

    # synthetic 제외 필터 + challenge_pass는 challenge로 집계
    _WHERE_BASE = """
        WHERE created_at >= :start
          AND created_at < :end
          AND status != 'synthetic'
    """

    try:
        # ── 기간 요약 ──
        summary_result = await db.execute(text(f"""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE status = 'pass') AS pass_count,
                COUNT(*) FILTER (WHERE status IN ('challenge', 'challenge_pass')) AS challenge_count,
                COUNT(*) FILTER (WHERE status = 'block') AS block_count,
                COUNT(*) FILTER (WHERE status = 'challenge_pass') AS challenge_pass_count,
                COUNT(*) FILTER (WHERE status = 'challenge') AS challenge_pending_count,
                AVG(solve_time_ms) FILTER (WHERE status = 'challenge_pass' AND solve_time_ms IS NOT NULL) AS avg_solve_time_ms
            FROM captcha_sessions
            {_WHERE_BASE}
        """), params)
        summary = summary_result.mappings().first()

        total = summary["total"] or 0
        pass_count = summary["pass_count"] or 0
        challenge_count = summary["challenge_count"] or 0
        block_count = summary["block_count"] or 0
        challenge_pass_count = summary["challenge_pass_count"] or 0
        challenge_pending_count = summary["challenge_pending_count"] or 0
        avg_solve_ms = round(summary["avg_solve_time_ms"] or 0)

        # ── 점수 분포 (선택 기간, synthetic 제외) ──
        dist_result = await db.execute(text(f"""
            SELECT
                FLOOR(LEAST(final_score, 0.9999) * 10)::int AS bucket,
                COUNT(*) AS cnt
            FROM captcha_sessions
            {_WHERE_BASE}
            GROUP BY bucket
            ORDER BY bucket
        """), params)
        dist_map = {row["bucket"]: row["cnt"] for row in dist_result.mappings()}
        score_distribution = [
            {"range": f"{i/10:.1f}-{(i+1)/10:.1f}", "count": dist_map.get(i, 0)}
            for i in range(10)
        ]

        # ── 추이 (period에 따라 집계 단위 변경) ──
        if period == "daily":
            trend_sql = f"""
                SELECT
                    created_at::date AS label,
                    COUNT(*) FILTER (WHERE status = 'pass') AS pass_count,
                    COUNT(*) FILTER (WHERE status IN ('challenge', 'challenge_pass')) AS challenge_count,
                    COUNT(*) FILTER (WHERE status = 'block') AS block_count
                FROM captcha_sessions
                {_WHERE_BASE}
                GROUP BY label
                ORDER BY label
            """
        elif period == "weekly":
            trend_sql = f"""
                SELECT
                    DATE_TRUNC('week', created_at)::date AS label,
                    COUNT(*) FILTER (WHERE status = 'pass') AS pass_count,
                    COUNT(*) FILTER (WHERE status IN ('challenge', 'challenge_pass')) AS challenge_count,
                    COUNT(*) FILTER (WHERE status = 'block') AS block_count
                FROM captcha_sessions
                {_WHERE_BASE}
                GROUP BY label
                ORDER BY label
            """
        else:  # monthly
            trend_sql = f"""
                SELECT
                    DATE_TRUNC('month', created_at)::date AS label,
                    COUNT(*) FILTER (WHERE status = 'pass') AS pass_count,
                    COUNT(*) FILTER (WHERE status IN ('challenge', 'challenge_pass')) AS challenge_count,
                    COUNT(*) FILTER (WHERE status = 'block') AS block_count
                FROM captcha_sessions
                {_WHERE_BASE}
                GROUP BY label
                ORDER BY label
            """

        trend_result = await db.execute(text(trend_sql), params)

        trend = []
        for row in trend_result.mappings():
            label_date = str(row["label"])
            if period == "weekly":
                # 주 시작일 표시 (예: "04-14~04-20")
                week_start = row["label"]
                week_end = week_start + timedelta(days=6)
                display = f"{week_start.strftime('%m-%d')}~{week_end.strftime('%m-%d')}"
            elif period == "monthly":
                display = row["label"].strftime("%Y-%m")
            else:
                display = row["label"].strftime("%m-%d")

            trend.append({
                "date": label_date,
                "display": display,
                "pass": row["pass_count"] or 0,
                "challenge": row["challenge_count"] or 0,
                "block": row["block_count"] or 0,
            })

        # 챌린지 통과율
        challenge_total = challenge_pass_count + challenge_pending_count
        challenge_pass_rate = round(
            challenge_pass_count / max(challenge_total, 1) * 100, 1
        )

        return {
            "period": period,
            "start_date": str(s_date),
            "end_date": str(e_date),
            "summary": {
                "total": total,
                "pass_count": pass_count,
                "challenge_count": challenge_count,
                "block_count": block_count,
                "pass_rate": round(pass_count / max(total, 1) * 100, 1),
                "challenge_rate": round(challenge_count / max(total, 1) * 100, 1),
                "block_rate": round(block_count / max(total, 1) * 100, 1),
            },
            "challenge_detail": {
                "total": challenge_total,
                "pass_count": challenge_pass_count,
                "pending_count": challenge_pending_count,
                "pass_rate": challenge_pass_rate,
                "avg_solve_time_ms": avg_solve_ms,
            },
            "score_distribution": score_distribution,
            "trend": trend,
        }
    except Exception as e:
        import logging
        logger = logging.getLogger("admin.captcha.stats")
        logger.error(f"[captcha/stats] 쿼리 실패: {type(e).__name__}: {e}")

        if "captcha_sessions" in str(e) or "relation" in str(e):
            return {
                "period": period,
                "start_date": str(s_date),
                "end_date": str(e_date),
                "summary": {
                    "total": 0, "pass_count": 0, "challenge_count": 0, "block_count": 0,
                    "pass_rate": 0, "challenge_rate": 0, "block_rate": 0,
                },
                "challenge_detail": {
                    "total": 0, "pass_count": 0, "pending_count": 0,
                    "pass_rate": 0, "avg_solve_time_ms": 0,
                },
                "score_distribution": [
                    {"range": f"{i/10:.1f}-{(i+1)/10:.1f}", "count": 0} for i in range(10)
                ],
                "trend": [],
            }
        raise


@router.get("/captcha/sessions", tags=["admin-captcha"])
async def list_captcha_sessions(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    status_filter: str | None = Query(default=None, alias="status"),
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """최근 캡챠 세션 로그 (페이지네이션)

    - page: 페이지 번호 (1부터)
    - size: 페이지 크기 (기본 20, 최대 100)
    - status: 필터 (pass / challenge / block)
    """
    try:
        where_clause = ""
        params: dict = {"limit": size, "offset": (page - 1) * size}

        if status_filter and status_filter in ("pass", "challenge", "block"):
            where_clause = "WHERE status = :status"
            params["status"] = status_filter

        # 전체 건수
        count_result = await db.execute(
            text(f"SELECT COUNT(*) AS cnt FROM captcha_sessions {where_clause}"),
            params,
        )
        total = count_result.scalar() or 0

        # 세션 목록
        rows_result = await db.execute(text(f"""
            SELECT
                id, trigger_type, client_ip::text AS client_ip,
                behavior_score, vector_score, lstm_score,
                final_score, status, attempt_count,
                solve_time_ms, is_correct,
                created_at
            FROM captcha_sessions
            {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset
        """), params)

        sessions = []
        for row in rows_result.mappings():
            sessions.append({
                "id": str(row["id"]),
                "trigger_type": row["trigger_type"],
                "client_ip": row["client_ip"],
                "behavior_score": row["behavior_score"],
                "vector_score": row["vector_score"],
                "lstm_score": row["lstm_score"],
                "final_score": row["final_score"],
                "status": row["status"],
                "attempt_count": row["attempt_count"],
                "solve_time_ms": row["solve_time_ms"],
                "is_correct": row["is_correct"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            })

        return {
            "sessions": sessions,
            "total": total,
            "page": page,
            "size": size,
            "total_pages": (total + size - 1) // size,
        }
    except Exception as e:
        import logging
        logging.getLogger("admin.captcha.sessions").error(
            f"[captcha/sessions] 쿼리 실패: {type(e).__name__}: {e}"
        )
        if "captcha_sessions" in str(e) or "relation" in str(e):
            return {
                "sessions": [],
                "total": 0,
                "page": page,
                "size": size,
                "total_pages": 0,
            }
        raise


@router.get("/captcha/sessions/{session_id}/images", tags=["admin-captcha"])
async def get_session_images(
    session_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """캡챠 세션에 사용된 이모지 + 실사 이미지를 프록시 URL로 반환

    - emojis: 이모지 이미지 목록 (카테고리 + 프록시 URL)
    - photos: 실사 사진 목록 (카테고리 + 프록시 URL)
    - answer_indices: 정답 위치 (photos 배열 내 인덱스)
    """
    import services.captcha_service as cs

    try:
        # 1) 세션 → captcha_set_id 조회
        session_result = await db.execute(text("""
            SELECT captcha_set_id FROM captcha_sessions WHERE id = :sid
        """), {"sid": session_id})
        session_row = session_result.mappings().first()

        if not session_row or not session_row["captcha_set_id"]:
            return {
                "session_id": session_id,
                "emojis": [],
                "photos": [],
                "answer_indices": [],
                "message": "이 세션에 연결된 캡챠 세트가 없습니다 (pass 세션일 수 있음)",
            }

        set_id = session_row["captcha_set_id"]

        # 2) captcha_sets → emoji_ids, photo_ids, answer_indices
        set_result = await db.execute(text("""
            SELECT emoji_ids, photo_ids, answer_indices
            FROM captcha_sets WHERE id = :set_id
        """), {"set_id": set_id})
        set_row = set_result.mappings().first()

        if not set_row:
            raise HTTPException(status_code=404, detail="캡챠 세트를 찾을 수 없습니다")

        emoji_ids = set_row["emoji_ids"] or []
        photo_ids = set_row["photo_ids"] or []
        answer_indices = set_row["answer_indices"] or []

        # 3) emoji_images 조회
        emojis = []
        if emoji_ids:
            emoji_result = await db.execute(text("""
                SELECT id, category, image_key
                FROM emoji_images
                WHERE id = ANY(:ids)
            """), {"ids": emoji_ids})
            emoji_map = {str(r["id"]): r for r in emoji_result.mappings()}

            for eid in emoji_ids:
                row = emoji_map.get(str(eid))
                if row:
                    token = await cs._create_image_token("captcha-emojis", row["image_key"])
                    emojis.append({
                        "id": str(row["id"]),
                        "category": row["category"],
                        "url": cs._build_proxy_url(token),
                    })

        # 4) real_photos 조회
        photos = []
        if photo_ids:
            photo_result = await db.execute(text("""
                SELECT id, category, image_key
                FROM real_photos
                WHERE id = ANY(:ids)
            """), {"ids": photo_ids})
            photo_map = {str(r["id"]): r for r in photo_result.mappings()}

            for pid in photo_ids:
                row = photo_map.get(str(pid))
                if row:
                    token = await cs._create_image_token("captcha-photos", row["image_key"])
                    photos.append({
                        "id": str(row["id"]),
                        "category": row["category"],
                        "url": cs._build_proxy_url(token),
                    })

        return {
            "session_id": session_id,
            "captcha_set_id": str(set_id),
            "emojis": emojis,
            "photos": photos,
            "answer_indices": answer_indices,
        }

    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger("admin.captcha.images").error(
            f"[captcha/sessions/images] 조회 실패: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"이미지 조회 실패: {e}")


# ── 이미지 관리 ──────────────────────────────────────

@router.get("/captcha/images", tags=["admin-captcha"])
async def list_captcha_images(
    image_type: str = Query(description="emoji 또는 photo"),
    category: str | None = Query(default=None, description="동물 카테고리 필터"),
    page: int = Query(default=1, ge=1),
    size: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """이모지 또는 실사 이미지 목록 조회 (카테고리별 필터 + 페이지네이션)"""
    import services.captcha_service as cs

    if image_type not in ("emoji", "photo"):
        raise HTTPException(status_code=400, detail="image_type은 'emoji' 또는 'photo'여야 합니다")

    try:
        table = "emoji_images" if image_type == "emoji" else "real_photos"
        bucket = "captcha-emojis" if image_type == "emoji" else "captcha-photos"

        where_parts = ["is_active = true"]
        params: dict = {"limit": size, "offset": (page - 1) * size}

        if category:
            where_parts.append("category = :category")
            params["category"] = category

        where_clause = "WHERE " + " AND ".join(where_parts)

        # 카테고리 목록 (필터 UI용)
        cat_result = await db.execute(text(f"""
            SELECT category, COUNT(*) AS cnt
            FROM {table}
            WHERE is_active = true
            GROUP BY category
            ORDER BY category
        """))
        categories = [
            {"category": r["category"], "count": r["cnt"]}
            for r in cat_result.mappings()
        ]

        # 전체 건수
        count_result = await db.execute(
            text(f"SELECT COUNT(*) AS cnt FROM {table} {where_clause}"),
            params,
        )
        total = count_result.scalar() or 0

        # 이미지 목록
        rows_result = await db.execute(text(f"""
            SELECT id, category, image_key, created_at
            FROM {table}
            {where_clause}
            ORDER BY category, created_at DESC
            LIMIT :limit OFFSET :offset
        """), params)

        images = []
        for row in rows_result.mappings():
            token = await cs._create_image_token(bucket, row["image_key"])
            images.append({
                "id": str(row["id"]),
                "category": row["category"],
                "image_key": row["image_key"],
                "url": cs._build_proxy_url(token),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            })

        return {
            "image_type": image_type,
            "categories": categories,
            "images": images,
            "total": total,
            "page": page,
            "size": size,
            "total_pages": (total + size - 1) // size,
        }

    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger("admin.captcha.images").error(
            f"[captcha/images] 조회 실패: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"이미지 목록 조회 실패: {e}")


@router.get("/captcha/images/{image_id}/sets", tags=["admin-captcha"])
async def get_image_sets(
    image_id: str,
    image_type: str = Query(description="emoji 또는 photo"),
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """특정 이미지가 사용된 캡챠 세트 목록 조회"""
    if image_type not in ("emoji", "photo"):
        raise HTTPException(status_code=400, detail="image_type은 'emoji' 또는 'photo'여야 합니다")

    try:
        # UUID 배열에 해당 이미지 ID가 포함된 세트 조회
        column = "emoji_ids" if image_type == "emoji" else "photo_ids"

        result = await db.execute(text(f"""
            SELECT id, emoji_ids, photo_ids, answer_indices,
                   use_count, is_active, created_by, created_at
            FROM captcha_sets
            WHERE :image_id = ANY({column})
            ORDER BY created_at DESC
            LIMIT 50
        """), {"image_id": image_id})

        sets = []
        for row in result.mappings():
            sets.append({
                "id": str(row["id"]),
                "emoji_count": len(row["emoji_ids"] or []),
                "photo_count": len(row["photo_ids"] or []),
                "answer_indices": row["answer_indices"] or [],
                "use_count": row["use_count"] or 0,
                "is_active": row["is_active"],
                "created_by": row["created_by"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            })

        return {
            "image_id": image_id,
            "image_type": image_type,
            "sets": sets,
            "total": len(sets),
        }

    except Exception as e:
        import logging
        logging.getLogger("admin.captcha.images").error(
            f"[captcha/images/sets] 조회 실패: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"세트 조회 실패: {e}")


@router.put("/captcha/sets/{set_id}/deactivate", tags=["admin-captcha"])
async def deactivate_captcha_set(
    set_id: str,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """캡챠 세트 비활성화 (is_active = false)"""
    try:
        result = await db.execute(text("""
            UPDATE captcha_sets
            SET is_active = false
            WHERE id = :set_id AND is_active = true
            RETURNING id
        """), {"set_id": set_id})
        await db.commit()

        updated = result.fetchone()
        if not updated:
            raise HTTPException(
                status_code=404,
                detail="세트를 찾을 수 없거나 이미 비활성화 상태입니다",
            )

        return {
            "set_id": set_id,
            "is_active": False,
            "message": f"캡챠 세트 {set_id[:8]}... 비활성화 완료",
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        import logging
        logging.getLogger("admin.captcha.sets").error(
            f"[captcha/sets/deactivate] 실패: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"세트 비활성화 실패: {e}")


@router.put("/captcha/images/{image_id}/deactivate", tags=["admin-captcha"])
async def deactivate_image(
    image_id: str,
    image_type: str = Query(description="emoji 또는 photo"),
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """이미지 비활성화 + 해당 이미지가 포함된 활성 세트 일괄 정지"""
    if image_type not in ("emoji", "photo"):
        raise HTTPException(status_code=400, detail="image_type은 'emoji' 또는 'photo'여야 합니다")

    try:
        table = "emoji_images" if image_type == "emoji" else "real_photos"
        column = "emoji_ids" if image_type == "emoji" else "photo_ids"

        # 1) 이미지 비활성화
        img_result = await db.execute(text(f"""
            UPDATE {table}
            SET is_active = false
            WHERE id = :image_id AND is_active = true
            RETURNING id, category
        """), {"image_id": image_id})
        img_row = img_result.fetchone()

        if not img_row:
            raise HTTPException(
                status_code=404,
                detail="이미지를 찾을 수 없거나 이미 비활성화 상태입니다",
            )

        # 2) 해당 이미지가 포함된 활성 세트 일괄 정지
        sets_result = await db.execute(text(f"""
            UPDATE captcha_sets
            SET is_active = false
            WHERE :image_id = ANY({column}) AND is_active = true
            RETURNING id
        """), {"image_id": image_id})
        deactivated_sets = [str(r[0]) for r in sets_result.fetchall()]

        await db.commit()

        return {
            "image_id": image_id,
            "image_type": image_type,
            "category": img_row[1],
            "deactivated_sets_count": len(deactivated_sets),
            "message": f"이미지 비활성화 완료. 연관 세트 {len(deactivated_sets)}개 정지됨",
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        import logging
        logging.getLogger("admin.captcha.images").error(
            f"[captcha/images/deactivate] 실패: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"이미지 비활성화 실패: {e}")


# ── 이미지 일괄 비활성화 ──────────────────────────────────────

@router.put("/captcha/images/batch-deactivate", tags=["admin-captcha"])
async def batch_deactivate_images(
    body: dict,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
):
    """여러 이미지 일괄 비활성화 + 관련 세트 연쇄 비활성화

    body: {"image_ids": ["uuid1", "uuid2", ...], "image_type": "emoji" | "photo"}
    """
    image_ids = body.get("image_ids", [])
    image_type = body.get("image_type", "emoji")

    if not image_ids:
        raise HTTPException(status_code=400, detail="image_ids가 비어있습니다")
    if image_type not in ("emoji", "photo"):
        raise HTTPException(status_code=400, detail="image_type은 'emoji' 또는 'photo'여야 합니다")

    try:
        table = "emoji_images" if image_type == "emoji" else "real_photos"
        column = "emoji_ids" if image_type == "emoji" else "photo_ids"

        # 1) 이미지 일괄 비활성화
        img_result = await db.execute(text(f"""
            UPDATE {table}
            SET is_active = false
            WHERE id = ANY(:ids) AND is_active = true
            RETURNING id
        """), {"ids": image_ids})
        deactivated_images = [str(r[0]) for r in img_result.fetchall()]

        # 2) 각 이미지가 포함된 활성 세트 일괄 정지
        total_sets_deactivated = 0
        for img_id in deactivated_images:
            sets_result = await db.execute(text(f"""
                UPDATE captcha_sets
                SET is_active = false
                WHERE :image_id = ANY({column}) AND is_active = true
                RETURNING id
            """), {"image_id": img_id})
            total_sets_deactivated += len(sets_result.fetchall())

        await db.commit()

        return {
            "deactivated_images": len(deactivated_images),
            "deactivated_sets": total_sets_deactivated,
            "message": f"이미지 {len(deactivated_images)}장 비활성화, 관련 세트 {total_sets_deactivated}개 정지",
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        import logging
        logging.getLogger("admin.captcha.images").error(
            f"[captcha/images/batch-deactivate] 실패: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail=f"일괄 비활성화 실패: {e}")


# ── 이모지 자동 생성 + 세트 생성 ──────────────────────────────

@router.post("/captcha/generate", tags=["admin-captcha"])
async def generate_captcha_images(
    body: dict,
    current_user: User = Depends(require_user),
):
    """GAN 이모지 생성 → CLIP 필터 → MinIO 업로드 → DB 등록 → 세트 조합 (비동기 실행)

    body: {
        "num_per_category": 30,   # 카테고리당 생성 수 (10~100)
        "num_sets": 50,           # 생성할 캡챠 세트 수
        "categories": "bear,cat,dog,elephant,fox,rabbit,lion,penguin,tiger"  # 선택
    }
    """
    import asyncio
    import subprocess

    num_per_category = body.get("num_per_category", 30)
    num_sets = body.get("num_sets", 50)
    cats = body.get("categories", "bear,cat,dog,elephant,fox,rabbit,lion,penguin,tiger")

    # 범위 검증
    if not (10 <= num_per_category <= 100):
        raise HTTPException(status_code=400, detail="num_per_category는 10~100 사이여야 합니다")
    if not (1 <= num_sets <= 500):
        raise HTTPException(status_code=400, detail="num_sets는 1~500 사이여야 합니다")

    # 이미 생성 중인지 확인 (Redis 락)
    lock_key = "captcha:generate:running"
    is_running = await redis_client.get(lock_key)
    if is_running:
        return {
            "status": "already_running",
            "message": "이미 생성이 진행 중입니다. 잠시 후 다시 시도해주세요.",
        }

    # 5분 락 설정
    await redis_client.setex(lock_key, 300, "1")
    await redis_client.set("captcha:generate:progress", "starting")

    async def run_generate():
        try:
            await redis_client.set("captcha:generate:progress", "generating")

            cmd = [
                "python", "/home/ubuntu/ganpipeline/generate_and_register.py",
                "--categories", cats,
                "--num_per_category", str(num_per_category),
                "--num_sets", str(num_sets),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd="/home/ubuntu/ganpipeline",
            )
            stdout, _ = await process.communicate()
            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            if process.returncode == 0:
                await redis_client.set("captcha:generate:progress", "done")
                await redis_client.setex("captcha:generate:last_result", 3600, output[-2000:])
            else:
                await redis_client.set("captcha:generate:progress", f"error: exit code {process.returncode}")
                await redis_client.setex("captcha:generate:last_result", 3600, output[-2000:])
        except Exception as e:
            await redis_client.set("captcha:generate:progress", f"error: {str(e)[:200]}")
        finally:
            await redis_client.delete(lock_key)

    # 백그라운드 실행
    asyncio.create_task(run_generate())

    return {
        "status": "started",
        "num_per_category": num_per_category,
        "num_sets": num_sets,
        "categories": cats,
        "message": f"이모지 생성 시작 (카테고리당 {num_per_category}장, 세트 {num_sets}개)",
    }


@router.get("/captcha/generate/status", tags=["admin-captcha"])
async def get_generate_status(
    current_user: User = Depends(require_user),
):
    """이모지 생성 진행 상태 조회"""
    progress = await redis_client.get("captcha:generate:progress")
    last_result = await redis_client.get("captcha:generate:last_result")

    if isinstance(progress, bytes):
        progress = progress.decode()
    if isinstance(last_result, bytes):
        last_result = last_result.decode()

    return {
        "progress": progress or "idle",
        "last_result": last_result,
    }
