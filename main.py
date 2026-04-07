import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import settings
from core.database import Base, engine
from routers import auth, captcha, chat, notifications, parties
# 상원: 관심사 저장/조회 API를 메인 앱에 등록하기 위해 라우터를 추가합니다.
from routers import user_interests  # 상원

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


@app.middleware("http")
async def log_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})


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
# 상원: 회원가입 뒤 관심사 저장 화면과 마이페이지가 같은 관심사 API를 사용할 수 있게 연결합니다.
app.include_router(user_interests.router, prefix="/api")  # 상원

ANIMAL_ASSET_DIR = Path(__file__).resolve().parent.parent / "animal"
if ANIMAL_ASSET_DIR.exists():
    app.mount(
        "/animal-assets",
        StaticFiles(directory=str(ANIMAL_ASSET_DIR)),
        name="animal-assets",
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}
