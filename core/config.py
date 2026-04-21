from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # DB
    DATABASE_URL: str

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60   # 60분으로 변경
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14
    CAPTCHA_JWT_SECRET: str = ""

    # CORS
    ALLOWED_ORIGINS: list[str]

    # Redis
    REDIS_URL: str
    REDIS_HOST: str = ""
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # Ollama
    OLLAMA_URL: str
    OLLAMA_MODEL: str = "exaone3.5:7.8b"
    OLLAMA_EMBED_MODEL : str = "nomic-embed-text"

    # GPU
    GPU_SERVER_URL: str

    # Email
    MAIL_USERNAME: str = ""
    MAIL_PASSWORD: str = ""
    MAIL_FROM: str = ""
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False
    EMAIL_AUTH_TTL_SECONDS: int = 180

    # Cookie
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"

    # Captcha
    CAPTCHA_PASS_THRESHOLD: float = 0.7
    CAPTCHA_CHALLENGE_THRESHOLD: float = 0.3
    CAPTCHA_SESSION_TTL_SECONDS: int = 120
    CAPTCHA_TOKEN_TTL_SECONDS: int = 300
    CAPTCHA_TOKEN_MAX_USES: int = 3
    CAPTCHA_MAX_ATTEMPTS: int = 5
    CAPTCHA_LOCK_SECONDS_1: int = 300    # 1회차 잠금: 5분
    CAPTCHA_LOCK_SECONDS_2: int = 600    # 2회차 잠금: 10분
    CAPTCHA_BAN_SECONDS: int = 86400
    CAPTCHA_WAIT_SECONDS: int = 30
    CAPTCHA_RATE_LIMIT_WINDOW_SECONDS: int = 60
    CAPTCHA_RATE_LIMIT_MAX_REQUESTS: int = 10
    CAPTCHA_MIN_SOLVE_SECONDS: float = 0.8
    # LSTM
    LSTM_ENABLED: bool = True
    LSTM_SHADOW_MODE: bool = True    # True면 로그만, final_score 미반영
    LSTM_WEIGHT: float = 0.7         # shadow 해제 후: rule 0.1 + KNN 0.2 + LSTM 0.7
    # MinIO
    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    PROFILE_MINIO_BUCKET: str 
    MINIO_EMOJI_BUCKET: str
    MINIO_PHOTO_BUCKET: str
    MINIO_SECURE: bool = False
    MINIO_PUBLIC_ENDPOINT: str = ""

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

    # 포트원
    PORTONE_API_SECRET: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
