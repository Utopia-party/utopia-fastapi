from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from models.report import Report
from services.notification_service import notify_user


def _resolve_report_result_label(action_result_code: str | None, status: str | None) -> str:
    action_code = (action_result_code or "NONE").strip().upper()
    status_code = (status or "").strip().upper()

    if action_code == "WARNING":
        return "경고 조치"
    if action_code == "PENALTY":
        return "제재 조치"
    if status_code == "REJECTED":
        return "신고가 기각되었어요"
    if status_code == "APPROVED":
        return "신고가 처리되었어요"

    return "처리 결과가 등록되었어요"


def _build_report_detail_message(report: Report) -> str:
    base = f"신고 대상: {report.target_snapshot_name or '알 수 없음'}"
    result_label = _resolve_report_result_label(
        report.action_result_code,
        report.status,
    )

    if report.admin_memo:
        return f"{base}\n처리 결과: {result_label}\n상세: {report.admin_memo}"

    return f"{base}\n처리 결과: {result_label}"


def _build_target_report_result_message(report: Report) -> str:
    result_label = _resolve_report_result_label(
        report.action_result_code,
        report.status,
    )
    base = "회원님에 대한 신고가 처리되었어요."

    if report.admin_memo:
        return f"{base}\n처리 결과: {result_label}\n상세: {report.admin_memo}"

    return f"{base}\n처리 결과: {result_label}"


async def notify_report_submitted(
    db: AsyncSession,
    *,
    report: Report,
) -> None:
    """
    신고자용: 신고 접수 완료
    """
    await notify_user(
        db=db,
        user_id=report.reporter_id,
        type="report",
        title="신고가 접수되었어요",
        message=f"{report.target_snapshot_name or '대상'}에 대한 신고가 접수되었어요.",
        reference_type="report",
        reference_id=report.id,
        metadata={
            "event_code": "REPORT_SUBMITTED",
            "report_id": str(report.id),
            "target_type": report.target_type,
            "target_id": str(report.target_id),
            "category": report.category,
        },
    )


async def notify_report_result_to_reporter(
    db: AsyncSession,
    *,
    report: Report,
) -> None:
    """
    신고자용: 신고 처리 완료 + 처리 결과 상세
    관리자 검토 완료 시점에 호출
    """
    await notify_user(
        db=db,
        user_id=report.reporter_id,
        type="report",
        title="신고 처리 결과가 안내되었어요",
        message=_build_report_detail_message(report),
        reference_type="report",
        reference_id=report.id,
        metadata={
            "event_code": "REPORT_RESOLVED",
            "report_id": str(report.id),
            "target_type": report.target_type,
            "target_id": str(report.target_id),
            "status": report.status,
            "action_result_code": report.action_result_code,
            "admin_memo": report.admin_memo,
        },
    )


async def notify_report_warning_to_target(
    db: AsyncSession,
    *,
    report: Report,
    warning_message: str | None = None,
) -> None:
    """
    피신고자용: 경고 알림
    관리자 검토 결과 경고 조치가 확정됐을 때 호출
    """
    await notify_user(
        db=db,
        user_id=report.target_id,
        type="report",
        title="운영 경고가 적용되었어요",
        message=warning_message or "신고 검토 결과 운영 경고가 적용되었어요.",
        reference_type="report",
        reference_id=report.id,
        metadata={
            "event_code": "REPORT_WARNING",
            "report_id": str(report.id),
            "status": report.status,
            "action_result_code": report.action_result_code,
        },
    )


async def notify_report_penalty_to_target(
    db: AsyncSession,
    *,
    report: Report,
    penalty_message: str | None = None,
    penalty_code: str | None = None,
) -> None:
    """
    피신고자용: 제재 확정 알림
    신고 누적/운영 정책 위반으로 실제 제재가 확정됐을 때 호출
    """
    await notify_user(
        db=db,
        user_id=report.target_id,
        type="report",
        title="이용 제재가 적용되었어요",
        message=penalty_message or "운영 정책 위반으로 이용 제재가 적용되었어요.",
        reference_type="report",
        reference_id=report.id,
        metadata={
            "event_code": "REPORT_PENALTY",
            "report_id": str(report.id),
            "status": report.status,
            "action_result_code": report.action_result_code,
            "penalty_code": penalty_code,
        },
    )


async def notify_report_result_to_target(
    db: AsyncSession,
    *,
    report: Report,
) -> None:
    """
    피신고자용: 신고 처리 결과 안내
    관리자 검토 완료 시점에 호출
    """
    await notify_user(
        db=db,
        user_id=report.target_id,
        type="report",
        title="신고 처리 결과가 안내되었어요",
        message=_build_target_report_result_message(report),
        reference_type="report",
        reference_id=report.id,
        metadata={
            "event_code": "REPORT_TARGET_RESOLVED",
            "report_id": str(report.id),
            "target_type": report.target_type,
            "target_id": str(report.target_id),
            "status": report.status,
            "action_result_code": report.action_result_code,
            "admin_memo": report.admin_memo,
        },
    )
