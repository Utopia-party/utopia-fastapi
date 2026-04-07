import time
import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.database import Base, engine
from routers import auth, captcha, chat, notifications, parties, user_interests

logging.basicConfig(level=logging.DEBUG)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(
    title="Party-Up API",
    description="파티업 백엔드 API",
    version="1.0.0",
    lifespan=lifespan,
)

# [중요] CORS 미들웨어를 먼저 등록하여 모든 요청에 대해 보안 헤더를 처리합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 테스트 완료 후 settings.ALLOWED_ORIGINS로 복구하세요.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 예외 처리 미들웨어 (WebSocket 제외 로직 포함)
@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    # 웹소켓 요청은 HTTP 미들웨어 로직을 타지 않도록 즉시 통과
    if request.headers.get("upgrade") == "websocket":
        return await call_next(request)
        
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

# 2. 응답 시간 측정 미들웨어 (WebSocket 제외 로직 포함)
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    if request.headers.get("upgrade") == "websocket":
        return await call_next(request)

    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if elapsed_ms >= 200:
        print(f"[TIMING] {elapsed_ms:7.0f}ms  {request.method} {request.url.path}")
    return response

# 라우터 등록 (prefix="/api" 유지)
app.include_router(auth.router, prefix="/api")
app.include_router(parties.router, prefix="/api")
app.include_router(notifications.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(captcha.router, prefix="/api", tags=["Captcha"])
app.include_router(user_interests.router, prefix="/api")

@app.get("/api/health")
async def health():
    return {"status": "ok"}
