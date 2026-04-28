from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database import get_db
from core.security import get_current_user
from models.mypage.trust_score import TrustScore
from models.report import Report, ReportEvidence
from models.user import User
from services.notifications.report_notification_service import (
    notify_report_penalty_to_target,
    notify_report_result_to_reporter,
    notify_report_result_to_target,
    notify_report_warning_to_target,
)
from services.report_storage_service import get_report_file_bytes


router = APIRouter(prefix="/admin/reports", tags=["admin-reports"])


ALLOWED_ADMIN_REPORT_STATUSES = {
    "PENDING",
    "IN_REVIEW",
    "APPROVED",
    "REJECTED",
    "접수",
    "검토중",
    "처리",
    "기각",
}


STATUS_TO_API = {
    "접수": "PENDING",
    "검토중": "IN_REVIEW",
    "처리": "APPROVED",
    "기각": "REJECTED",
    "PENDING": "PENDING",
    "IN_REVIEW": "IN_REVIEW",
    "APPROVED": "APPROVED",
    "REJECTED": "REJECTED",
}


ACTION_RESULT_BY_STATUS = {
    "PENDING": "NONE",
    "IN_REVIEW": "NONE",
    "APPROVED": "WARNING",
    "REJECTED": "NONE",
}


AUTO_REPORT_PENALTY = {
    "PROFANITY": -1.0,
    "SPAM": -5.0,
    "SCAM": -5.0,
}


def normalize_admin_report_status(status: str) -> str:
    normalized = status.strip()

    if normalized not in ALLOWED_ADMIN_REPORT_STATUSES:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않은 신고 상태입니다.",
        )

    return STATUS_TO_API[normalized]


def build_admin_report_response(report: Report) -> dict[str, Any]:
    return {
        "id": str(report.id),
        "type": report.target_type,
        "target_type": report.target_type,
        "target": report.target_snapshot_name or str(report.target_id),
        "target_id": str(report.target_id),
        "target_snapshot_name": report.target_snapshot_name,
        "reason": report.category,
        "category": report.category,
        "status": report.status,
        "content": report.description,
        "description": report.description,
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "updated_at": report.updated_at.isoformat() if report.updated_at else None,
        "reporter_id": str(report.reporter_id),
        "reporter_nickname": None,
        "action_result_code": report.action_result_code,
        "admin_memo": report.admin_memo,
        "reviewed_by": str(report.reviewed_by) if report.reviewed_by else None,
        "reviewed_at": report.reviewed_at.isoformat() if report.reviewed_at else None,
        "evidences": [
            {
                "id": str(evidence.id),
                "object_key": evidence.object_key,
                "original_filename": evidence.original_filename,
                "content_type": evidence.content_type,
                "file_size": evidence.file_size,
                "created_at": evidence.created_at.isoformat()
                if evidence.created_at
                else None,
                "url": f"/api/admin/reports/evidences/{evidence.id}/file",
            }
            for evidence in report.evidences
        ],
    }


def _resolve_auto_report_penalty(report: Report) -> float:
    category = (report.category or "").strip().upper()
    return AUTO_REPORT_PENALTY.get(category, -1.0)


def _build_target_warning_message(report: Report, penalty: float, new_score: float) -> str:
    target_name = report.target_snapshot_name or "회원님의 계정"
    return (
        f"{target_name} 관련 신고가 승인되어 신뢰도 {abs(penalty):.1f}점이 차감되었어요. "
        f"현재 신뢰도는 {new_score:.1f}점입니다."
    )


async def _apply_report_penalty(
    db: AsyncSession,
    *,
    report: Report,
    reviewer_id: UUID,
) -> tuple[float | None, int | None]:
    if report.target_type != "USER":
        return None, None

    target_user = await db.get(User, report.target_id)
    if not target_user:
        return None, None

    penalty = _resolve_auto_report_penalty(report)
    previous_score = (
        float(target_user.trust_score)
        if target_user.trust_score is not None
        else 36.5
    )
    new_score = max(0.0, round(previous_score + penalty, 1))
    target_user.trust_score = new_score

    warn_count = (target_user.chat_warn_count or 0) + 1
    target_user.chat_warn_count = warn_count

    if new_score <= 0 or warn_count >= 4:
        target_user.is_active = False
        target_user.banned_until = None
        report.action_result_code = "PENALTY"
    elif new_score < 10 or warn_count >= 3:
        target_user.is_active = False
        target_user.banned_until = datetime.now(timezone.utc) + timedelta(days=30)
        report.action_result_code = "PENALTY"
    else:
        report.action_result_code = "WARNING"

    db.add(
        TrustScore(
            user_id=target_user.id,
            previous_score=previous_score,
            new_score=new_score,
            change_amount=round(new_score - previous_score, 1),
            reason=f"신고 승인: {report.category}",
            reference_id=report.id,
            created_by=reviewer_id,
        )
    )

    report.admin_memo = (
        f"자동 신뢰도 차감 {abs(penalty):.1f}점 적용 / 현재 {new_score:.1f}점"
    )
    return new_score, warn_count


async def ensure_admin_can_manage_reports(current_user: User) -> None:
    role = getattr(current_user, "role", None)
    is_admin = getattr(current_user, "is_admin", False)

    if role == "admin" or is_admin:
        return

    raise HTTPException(
        status_code=403,
        detail="관리자 권한이 필요합니다.",
    )


@router.get("")
async def list_admin_reports(
    keyword: str | None = Query(default=None),
    type: str | None = Query(default=None),
    date_from: date | None = Query(default=None),
    date_to: date | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await ensure_admin_can_manage_reports(current_user)

    query = (
        select(Report)
        .options(selectinload(Report.evidences))
        .order_by(Report.created_at.desc())
    )

    conditions = []

    if type:
        conditions.append(Report.target_type == type.upper().strip())

    if keyword:
        value = f"%{keyword.strip()}%"
        conditions.append(
            or_(
                Report.target_snapshot_name.ilike(value),
                Report.category.ilike(value),
                Report.description.ilike(value),
                Report.status.ilike(value),
                Report.target_type.ilike(value),
            )
        )

    if date_from:
        conditions.append(
            Report.created_at >= datetime.combine(date_from, time.min)
        )

    if date_to:
        conditions.append(
            Report.created_at <= datetime.combine(date_to, time.max)
        )

    if conditions:
        query = query.where(and_(*conditions))

    result = await db.execute(query)
    reports = result.scalars().unique().all()

    return [build_admin_report_response(report) for report in reports]


@router.patch("/{report_id}")
async def update_admin_report_status(
    report_id: UUID,
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await ensure_admin_can_manage_reports(current_user)

    next_status_raw = payload.get("status")

    if not isinstance(next_status_raw, str):
        raise HTTPException(
            status_code=400,
            detail="status 값이 필요합니다.",
        )

    next_status = normalize_admin_report_status(next_status_raw)

    result = await db.execute(
        select(Report)
        .options(selectinload(Report.evidences))
        .where(Report.id == report_id)
    )
    report = result.scalar_one_or_none()

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="신고를 찾을 수 없습니다.",
        )

    previous_status = report.status
    report.status = next_status
    report.reviewed_by = current_user.id
    report.reviewed_at = datetime.now(timezone.utc)
    report.action_result_code = ACTION_RESULT_BY_STATUS[next_status]

    new_score: float | None = None
    warn_count: int | None = None
    if previous_status != "APPROVED" and next_status == "APPROVED":
        new_score, warn_count = await _apply_report_penalty(
            db,
            report=report,
            reviewer_id=current_user.id,
        )

    await db.commit()

    result = await db.execute(
        select(Report)
        .options(selectinload(Report.evidences))
        .where(Report.id == report_id)
    )
    updated_report = result.scalar_one()

    if next_status == "APPROVED":
        await notify_report_result_to_reporter(
            db,
            report=updated_report,
        )
        if updated_report.target_type == "USER":
            await notify_report_result_to_target(
                db,
                report=updated_report,
            )
        if updated_report.target_type == "USER" and new_score is not None:
            message = _build_target_warning_message(
                updated_report,
                _resolve_auto_report_penalty(updated_report),
                new_score,
            )
            if updated_report.action_result_code == "PENALTY":
                await notify_report_penalty_to_target(
                    db,
                    report=updated_report,
                    penalty_message=message,
                    penalty_code=(
                        "PERMANENT_BAN"
                        if new_score <= 0 or (warn_count or 0) >= 4
                        else "TEMP_BAN_30_DAYS"
                    ),
                )
            else:
                await notify_report_warning_to_target(
                    db,
                    report=updated_report,
                    warning_message=message,
                )
    elif next_status == "REJECTED":
        await notify_report_result_to_reporter(
            db,
            report=updated_report,
        )
        if updated_report.target_type == "USER":
            await notify_report_result_to_target(
                db,
                report=updated_report,
            )

    return build_admin_report_response(updated_report)


@router.get("/evidences/{evidence_id}/file")
async def get_admin_report_evidence_file(
    evidence_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await ensure_admin_can_manage_reports(current_user)

    result = await db.execute(
        select(ReportEvidence)
        .join(Report, ReportEvidence.report_id == Report.id)
        .where(ReportEvidence.id == evidence_id)
    )
    evidence = result.scalar_one_or_none()

    if evidence is None:
        raise HTTPException(
            status_code=404,
            detail="증빙 파일을 찾을 수 없습니다.",
        )

    file_bytes, content_type = get_report_file_bytes(evidence.object_key)
    filename = evidence.original_filename or evidence.object_key.rsplit("/", 1)[-1]

    return Response(
        content=file_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}",
            "Cache-Control": "private, max-age=60",
        },
    )
