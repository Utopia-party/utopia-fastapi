from fastapi import APIRouter, Request
from pydantic import BaseModel
import logging
import asyncio
import redis.asyncio as redis

from core.config import settings

router = APIRouter()
logger = logging.getLogger("search-api")


# ─────────────────────────────────────────────
# Redis 설정 및 상/변수 정의
# ─────────────────────────────────────────────

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

TRENDING_KEY = "search:trending"
DECAY_INTERVAL_SECONDS = 10 * 60  # 10분마다 점수 반감

# Rate Limit 설정
# 동일 IP가 동일한 키워드를 검색할 때 점수 인정을 막는 쿨타임(초)
SEARCH_RATE_LIMIT_TTL = 60  


# ─────────────────────────────────────────────
# Pydantic 모델
# ─────────────────────────────────────────────

class RecordSearchRequest(BaseModel):
    keyword: str


# ─────────────────────────────────────────────
# 유틸리티 (IP 추출)
# ─────────────────────────────────────────────

def get_client_ip(request: Request) -> str:
    """
    운영에서 프록시(Nginx/ALB/Cloudflare) 뒤에 있다면
    해당 프록시를 신뢰하는 구성일 때만 X-Forwarded-For를 사용하세요.
    """
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()

    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


# ─────────────────────────────────────────────
# 백그라운드 태스크 (Score Decay)
# ─────────────────────────────────────────────

async def decay_trending_scores():
    """
    과거 검색어가 영원히 상위권에 머무는 것을 방지하기 위해,
    주기적으로 Redis Lua 스크립트를 실행하여 모든 검색어의 점수를 절반으로 깎습니다.
    점수가 1 미만으로 떨어지면 리스트에서 제거하여 메모리를 관리합니다.
    """
    script = """
    local elements = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
    for i=1, #elements, 2 do
        local member = elements[i]
        local score = tonumber(elements[i+1])
        local new_score = score / 2
        
        if new_score < 1 then
            redis.call('ZREM', KEYS[1], member)
        else
            redis.call('ZADD', KEYS[1], new_score, member)
        end
    end
    return true
    """
    
    while True:
        await asyncio.sleep(DECAY_INTERVAL_SECONDS)
        try:
            await redis_client.eval(script, 1, TRENDING_KEY)
            logger.info("Trending scores decayed successfully.")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error during decay_trending_scores: {e}")

# FastAPI 라우터가 시작될 때 백그라운드 루프 실행
# (만약 메인 app의 lifespan을 사용 중이시라면, 이 함수를 메인 파일의 lifespan으로 옮겨주세요)
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(decay_trending_scores())


# ─────────────────────────────────────────────
# API 엔드포인트: 실시간 인기 검색어 조회
# ─────────────────────────────────────────────

@router.get("/api/search/trending")
async def get_trending(limit: int = 5):
    try:
        # ZREVRANGE: 점수 내림차순(가장 높은 점수부터)으로 상위 limit 개수 조회
        keywords = await redis_client.zrevrange(TRENDING_KEY, 0, limit - 1)
        return {
            "success": True, 
            "keywords": keywords
        }
    except Exception as e:
        logger.exception("Failed to fetch trending keywords")
        return {
            "success": False, 
            "keywords": []
        }


# ─────────────────────────────────────────────
# API 엔드포인트: 검색어 기록 (Rate Limiting 적용)
# ─────────────────────────────────────────────

@router.post("/api/search/record")
async def record_search(payload: RecordSearchRequest, request: Request):
    # 1. IP 추출 및 키워드 정규화
    ip = get_client_ip(request)
    keyword = payload.keyword.strip()
    
    # 2. 의미 없는 단어나 너무 짧은 검색어 방지
    if not keyword or len(keyword) < 2:
        return {"success": False, "message": "Keyword too short or empty"}
        
    try:
        # 3. Rate Limiting 적용 (어뷰징 방지)
        # 키 형식: search:rl:{IP}:{검색어}
        rl_key = f"search:rl:{ip}:{keyword}"
        
        # NX=True: 키가 존재하지 않을 때만 설정(SET) 성공 (1 반환)
        # EX: 지정된 초(60초) 후 만료
        is_allowed = await redis_client.set(rl_key, "1", ex=SEARCH_RATE_LIMIT_TTL, nx=True)
        
        if not is_allowed:
            # 이미 60초 이내에 동일한 IP로 동일한 키워드를 검색한 경우
            # 에러를 던지면 프론트엔드 UX가 망가지므로, 성공으로 처리하되 점수는 올리지 않음 (Silent Drop)
            return {
                "success": True, 
                "keyword": keyword,
                "note": "rate_limited_ignored" 
            }

        # 4. 검증을 통과한 경우에만 ZINCRBY로 점수 1 증가
        await redis_client.zincrby(TRENDING_KEY, 1, keyword)
        return {
            "success": True, 
            "keyword": keyword
        }
    except Exception as e:
        logger.exception("Failed to record search keyword")
        return {
            "success": False, 
            "message": "Internal server error"
        }