"""
빠른매칭 학습 라벨링 태스크

1. label_quick_match_training_events : Celery beat 하루 1회 실행
   - 빠른매칭 가입 후 30일 이상 지난 pending 이벤트 검사
   - 유지/정산/신고/강퇴 조건에 따라 success, failed, excluded 라벨 확정
   - 라벨 확정 후 빠른매칭 학습 통계 테이블 재집계

2. _run_label_quick_match_training_events : 수동 실행/테스트용 async helper
"""

import asyncio
from datetime import datetime, timezone

from core.celery_app import celery_app
from core.database import AsyncSessionLocal

# SQLAlchemy relationship 문자열 해석을 위해 task 단독 실행 시에도 모델을 먼저 등록한다.
from models.notification import Notification  # noqa: F401
from models.party import Party, PartyMember, Service  # noqa: F401
from models.payment import Payment  # noqa: F401
from models.quick_match.candidate import QuickMatchCandidate  # noqa: F401
from models.quick_match.request import QuickMatchRequest  # noqa: F401
from models.quick_match.result import QuickMatchResult  # noqa: F401
from models.quick_match.training_events import QuickMatchTrainingEvent  # noqa: F401
from models.quick_match.training_stats import QuickMatchTrainingStat  # noqa: F401
from models.report import Report  # noqa: F401
from models.user import User  # noqa: F401

from services.quick_match.training_event_service import label_retained_successes
from services.quick_match.training_stats_service import rebuild_training_stats


DEFAULT_RETENTION_DAYS = 30


async def _run_label_quick_match_training_events(
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> dict:
    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        result = await label_retained_successes(
            db,
            retention_days=retention_days,
        )
        stats_result = await rebuild_training_stats(db)
        await db.commit()

    return {
        "retention_days": retention_days,
        "success_count": result.get("success", 0),
        "failed_count": result.get("failed", 0),
        "pending_count": result.get("pending", 0),
        "excluded_count": result.get("excluded", 0),
        "stats_created_count": stats_result.get("created_count", 0),
        "stats_event_sample_count": stats_result.get("event_sample_count", 0),
        "stats_generated_at": stats_result.get("generated_at"),
        "run_at": now.isoformat(),
    }


@celery_app.task(
    name="tasks.quick_match_training_label.label_quick_match_training_events",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
)
def label_quick_match_training_events(
    self,
    retention_days: int = DEFAULT_RETENTION_DAYS,
):
    try:
        return asyncio.run(
            _run_label_quick_match_training_events(
                retention_days=retention_days,
            )
        )
    except Exception as exc:
        raise self.retry(exc=exc)
