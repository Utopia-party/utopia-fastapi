from sqlalchemy.ext.asyncio import AsyncSession
from models.admin import ActivityLog

async def create_activity_log(
    db: AsyncSession,
    user_id,
    action: str,
    description: str = None,
    metadata: dict = None,
    target_id=None,
    ip_address: str = None,
    user_agent: str = None,
):
    log = ActivityLog(
        actor_user_id=user_id,
        action_type=action,
        description=description,
        extra_metadata=metadata or {},
        target_id=target_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    db.add(log)