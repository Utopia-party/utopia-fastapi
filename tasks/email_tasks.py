import asyncio

from fastapi_mail import ConnectionConfig, FastMail, MessageSchema

from core.config import settings
from core.celery_app import celery_app


conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_STARTTLS=settings.MAIL_STARTTLS,
    MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
    USE_CREDENTIALS=True,
)


@celery_app.task(
    name="tasks.email_tasks.send_email_task",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
)
def send_email_task(
    self,
    *,
    email: str,
    subject: str,
    body: str,
) -> None:
    try:
        message = MessageSchema(
            subject=subject,
            recipients=[email],
            body=body,
            subtype="plain",
        )

        fm = FastMail(conf)
        asyncio.run(fm.send_message(message))

    except Exception as exc:
        raise self.retry(exc=exc)