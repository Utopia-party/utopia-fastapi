from __future__ import annotations

import asyncio

from core.celery_app import celery_app
from core.database import AsyncSessionLocal
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
from services.quick_match.training_stats_service import rebuild_training_stats


async def _run_rebuild_quick_match_training_stats() -> dict:
    async with AsyncSessionLocal() as db:
        result = await rebuild_training_stats(db)
        await db.commit()
        return result


@celery_app.task(
    name="tasks.quick_match_training_stats.rebuild_quick_match_training_stats",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
)
def rebuild_quick_match_training_stats(self):
    try:
        return asyncio.run(_run_rebuild_quick_match_training_stats())
    except Exception as exc:
        raise self.retry(exc=exc)
