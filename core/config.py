from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DB
    DATABASE_URL: str

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    ALLOWED_ORIGINS: list[str]

    # Redis - REDIS_URL만 사용, 나머지는 .env 호환용으로만 선언
    REDIS_URL: str
    REDIS_HOST: str = ""       # redis_client.py에서 미사용, .env 호환용
    REDIS_PORT: int = 6379     # redis_client.py에서 미사용, .env 호환용
    REDIS_DB: int = 0          # redis_client.py에서 미사용, .env 호환용
    REDIS_PASSWORD: str = ""   # redis_client.py에서 미사용, .env 호환용

    # Ollama
    OLLAMA_URL: str
    OLLAMA_MODEL: str = "exaone3.5:7.8b"

    # GPU
    GPU_SERVER_URL: str

    # Email
    MAIL_USERNAME: str = ""
    MAIL_PASSWORD: str = ""
    MAIL_FROM: str = ""
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_STARTTLS: bool = True   # .env 호환용
    MAIL_SSL_TLS: bool = False   # .env 호환용
    EMAIL_AUTH_TTL_SECONDS: int = 180

    # Cookie
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"

    # OAuth - Google
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = ""

    # OAuth - Kakao
    KAKAO_REST_API_KEY: str = ""
    KAKAO_CLIENT_SECRET: str = ""
    KAKAO_REDIRECT_URI: str = ""

    # OAuth - Naver
    NAVER_CLIENT_ID: str = ""
    NAVER_CLIENT_SECRET: str = ""
    NAVER_REDIRECT_URI: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
