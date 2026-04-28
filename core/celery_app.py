from celery import Celery

from core.config import settings


broker_url = settings.CELERY_BROKER_URL or settings.REDIS_URL
result_backend = settings.CELERY_RESULT_BACKEND or settings.REDIS_URL


celery_app = Celery(
    "partyup",
    broker=broker_url,
    backend=result_backend,
    include=[
        "tasks.email_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)