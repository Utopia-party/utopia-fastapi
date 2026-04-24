from __future__ import annotations

from datetime import date, datetime, time
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
from models.report import Report, ReportEvidence
from models.user import User
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
    "APPROVED": "NO_ACTION",
    "REJECTED": "NO_ACTION",
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


async def ensure_admin_can_manage_reports(current_user: User) -> None:
    """
    TODO:
    기존 프로젝트의 관리자 권한 dependency가 있으면 이 함수 대신 그걸 연결하는 게 가장 좋습니다.

    현재는 User.role == "admin" 또는 User.is_admin == True 기준으로 검사합니다.
    프로젝트의 실제 관리자 권한 구조가 admin_roles.can_manage_reports라면 이 부분은 교체해야 합니다.
    """
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

    report.status = next_status
    report.reviewed_by = current_user.id
    report.reviewed_at = datetime.now()

    if next_status in ACTION_RESULT_BY_STATUS:
        report.action_result_code = ACTION_RESULT_BY_STATUS[next_status]

    await db.commit()

    result = await db.execute(
        select(Report)
        .options(selectinload(Report.evidences))
        .where(Report.id == report_id)
    )
    updated_report = result.scalar_one()

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