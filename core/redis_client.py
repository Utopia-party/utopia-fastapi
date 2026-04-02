import redis.asyncio as aioredis
from core.config import settings

# FastAPI async 환경에서 동기 redis는 event loop를 블로킹함
redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
