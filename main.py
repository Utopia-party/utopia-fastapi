from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.database import Base, engine
from routers import auth, captcha, chat, notifications, parties


# ✅ Fix: @app.on_event("startup") deprecated → lifespan으로 교체
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 실행 시 테이블 없을 경우 DB 자동생성
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="Party-Up API",
    description="파티업 백엔드 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(parties.router, prefix="/api")
app.include_router(notifications.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(captcha.router, prefix="/api", tags=["Captcha"])

#도상원
ANIMAL_ASSET_DIR = Path(__file__).resolve().parent.parent / "animal"
if ANIMAL_ASSET_DIR.exists():
    app.mount(
        "/animal-assets",
        StaticFiles(directory=str(ANIMAL_ASSET_DIR)),
        name="animal-assets",
    )
#도상원


# 헬스체크
@app.get("/api/health")
async def health():
    return {"status": "ok"}
