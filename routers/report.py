from __future__ import annotations

from collections import Counter
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database import get_db
from core.security import get_current_user
from models.report import Report, ReportEvidence
from models.user import User
from schemas.report import ReportResponse, ReportSummaryResponse
from services.notifications.report_notification_service import notify_report_submitted
from services.report_storage_service import upload_report_file
from services.report_target_service import resolve_target_snapshot_name


router = APIRouter(prefix="/reports", tags=["reports"])

ALLOWED_TARGET_TYPE = "USER"
ALLOWED_CATEGORIES = {"PROFANITY", "SCAM", "SPAM"}
ALLOWED_STATUSES = {"PENDING", "IN_REVIEW", "APPROVED", "REJECTED"}

MAX_REPORT_FILE_COUNT = 5


async def resolve_report_target_user_id(
    db: AsyncSession,
    target_identifier: str | None,
) -> UUID:
    if not target_identifier:
        raise HTTPException(
            status_code=400,
            detail="사용자 신고는 닉네임 또는 이메일이 필요합니다.",
        )

    value = target_identifier.strip()
    if not value:
        raise HTTPException(
            status_code=400,
            detail="사용자 신고는 닉네임 또는 이메일이 필요합니다.",
        )

    result = await db.execute(
        select(User.id).where(
            or_(User.nickname == value, User.email == value)
        )
    )
    resolved_id = result.scalar_one_or_none()

    if resolved_id is None:
        raise HTTPException(
            status_code=404,
            detail="신고 대상을 찾을 수 없습니다.",
        )

    return resolved_id


@router.post("", response_model=ReportResponse, status_code=status.HTTP_201_CREATED)
async def create_report(
    category: Annotated[str, Form(...)],
    description: Annotated[str, Form(...)],
    target_identifier: Annotated[str, Form(...)],
    files: list[UploadFile] | None = File(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    uploaded_object_keys: list[str] = []

    try:
        category = category.upper().strip()
        description = description.strip()
        target_identifier = target_identifier.strip()

        if category not in ALLOWED_CATEGORIES:
            raise HTTPException(
                status_code=400,
                detail="유효하지 않은 category 입니다.",
            )

        if not description:
            raise HTTPException(
                status_code=400,
                detail="신고 내용을 입력해주세요.",
            )

        if files and len(files) > MAX_REPORT_FILE_COUNT:
            raise HTTPException(
                status_code=400,
                detail=f"증빙 파일은 최대 {MAX_REPORT_FILE_COUNT}개까지 업로드할 수 있습니다.",
            )

        resolved_target_id = await resolve_report_target_user_id(
            db=db,
            target_identifier=target_identifier,
        )

        if resolved_target_id == current_user.id:
            raise HTTPException(
                status_code=400,
                detail="본인 계정은 신고할 수 없습니다.",
            )

        snapshot_name = await resolve_target_snapshot_name(
            db,
            ALLOWED_TARGET_TYPE,
            resolved_target_id,
        )
        if snapshot_name is None:
            raise HTTPException(
                status_code=404,
                detail="신고 대상을 찾을 수 없습니다.",
            )

        report = Report(
            reporter_id=current_user.id,
            target_type=ALLOWED_TARGET_TYPE,
            target_id=resolved_target_id,
            target_snapshot_name=snapshot_name,
            category=category,
            description=description,
            status="PENDING",
            action_result_code="NONE",
        )

        db.add(report)
        await db.flush()

        evidence_rows: list[ReportEvidence] = []

        if files:
            for file in files:
                if not file.filename:
                    continue

                uploaded = await upload_report_file(
                    file=file,
                    report_id=str(report.id),
                )

                uploaded_object_keys.append(uploaded["object_key"])

                evidence = ReportEvidence(
                    report_id=report.id,
                    object_key=uploaded["object_key"],
                    original_filename=uploaded.get("original_filename"),
                    content_type=uploaded.get("content_type"),
                    file_size=uploaded.get("file_size"),
                )
                evidence_rows.append(evidence)

            if evidence_rows:
                db.add_all(evidence_rows)
                report.evidence_key = evidence_rows[0].object_key

        await db.commit()

        result = await db.execute(
            select(Report)
            .options(selectinload(Report.evidences))
            .where(Report.id == report.id)
        )
        created_report = result.scalar_one()

        await notify_report_submitted(
            db=db,
            report=created_report,
        )

        return created_report

    except HTTPException:
        await db.rollback()
        raise

    except Exception:
        await db.rollback()
        raise


@router.get("", response_model=list[ReportResponse])
async def list_my_reports(
    status_filter: str | None = Query(default=None, alias="status"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = (
        select(Report)
        .options(selectinload(Report.evidences))
        .where(Report.reporter_id == current_user.id)
        .order_by(Report.created_at.desc())
    )

    if status_filter:
        normalized_status = status_filter.upper().strip()
        if normalized_status not in ALLOWED_STATUSES:
            raise HTTPException(
                status_code=400,
                detail="유효하지 않은 status 입니다.",
            )
        query = query.where(Report.status == normalized_status)

    result = await db.execute(query)
    reports = result.scalars().unique().all()

    return list(reports)


@router.get("/summary", response_model=ReportSummaryResponse)
async def get_my_report_summary(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Report.status, func.count(Report.id))
        .where(Report.reporter_id == current_user.id)
        .group_by(Report.status)
    )

    counts = Counter(
        {
            report_status: count
            for report_status, count in result.all()
        }
    )

    return ReportSummaryResponse(
        pending=counts.get("PENDING", 0),
        in_review=counts.get("IN_REVIEW", 0),
        approved=counts.get("APPROVED", 0),
        rejected=counts.get("REJECTED", 0),
    )