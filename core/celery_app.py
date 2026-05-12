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
        "tasks.party_trust_bonus",
        "tasks.payment_deadline",
        "tasks.quick_match_training_label",
        "tasks.quick_match_training_stats",
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
    beat_schedule={
        # 매일 00:00 KST(= 15:00 UTC)에 장기 파티 신뢰도 보너스 지급
        "party-trust-bonus-daily": {
            "task": "tasks.party_trust_bonus.run_party_trust_bonus",
            "schedule": 86400,  # 24h (초 단위)
        },
        # 매 10분마다 결제 마감일 초과 미결제 멤버 노쇼 처리
        "payment-deadline-check": {
            "task": "tasks.payment_deadline.check_payment_deadline",
            "schedule": 600,  # 10분
        },
        # 매일 1회 빠른매칭 pending 학습 이벤트 라벨 확정 및 통계분석
        "quick-match-training-label": {
            "task": "tasks.quick_match_training_label.label_quick_match_training_events",
            "schedule": 86400,  # 24시간
        },
    },
)
