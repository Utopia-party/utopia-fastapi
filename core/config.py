from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DB
    DATABASE_URL: str

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    CAPTCHA_JWT_SECRET: str = ""

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

    # Captcha - 1차 행동 기반 캡챠
    CAPTCHA_PASS_THRESHOLD: float = 0.7
    CAPTCHA_CHALLENGE_THRESHOLD: float = 0.3
    CAPTCHA_SESSION_TTL_SECONDS: int = 120
    CAPTCHA_TOKEN_TTL_SECONDS: int = 300
    CAPTCHA_TOKEN_MAX_USES: int = 3
    CAPTCHA_MAX_ATTEMPTS: int = 5
    CAPTCHA_LOCK_SECONDS: int = 1800
    CAPTCHA_BAN_SECONDS: int = 86400
    CAPTCHA_WAIT_SECONDS: int = 30
    CAPTCHA_RATE_LIMIT_WINDOW_SECONDS: int = 60
    CAPTCHA_RATE_LIMIT_MAX_REQUESTS: int = 10
    CAPTCHA_MIN_SOLVE_SECONDS: float = 0.8

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9001"
    MINIO_ACCESS_KEY: str = "admin1234"
    MINIO_SECRET_KEY: str = "partyup1234"
    MINIO_SECURE: bool = False
    MINIO_EMOJI_BUCKET: str = "captcha-emojis"
    MINIO_PHOTO_BUCKET: str = "captcha-photos"

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
