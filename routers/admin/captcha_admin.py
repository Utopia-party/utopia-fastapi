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


@router.get("/payments", response_model=AdminPaymentListOut)
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

@router.get("/moderation/chat-trend", response_model=list[dict])