"""
Core Configuration Module
=========================
Centralizes all application settings using Pydantic BaseSettings.
Environment variables override defaults.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator


class Settings(BaseSettings):
    # ─── Application ──────────────────────────────────────
    APP_NAME: str = "GEoN Risk Assessment Platform"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    DEBUG: bool = False
    SECRET_KEY: str = "change-this-in-production-use-32-char-minimum"
    API_V1_PREFIX: str = "/api/v1"

    # ─── Server ───────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # ─── Database ─────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgresuser:bjS6iZJ2gB6JCSm1OCtb3TNLEMWywDWK@dpg-d8cbv6jtqb8s738e0ufg-a.oregon-postgres.render.com/postgresdb_j3yf"
    DATABASE_URL_SYNC: str = "postgresql://postgresuser:bjS6iZJ2gB6JCSm1OCtb3TNLEMWywDWK@dpg-d8cbv6jtqb8s738e0ufg-a.oregon-postgres.render.com/postgresdb_j3yf"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # ─── Redis ────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600  # 1 hour default
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ─── JWT ──────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ─── Rate Limiting ────────────────────────────────────
    FREE_TIER_DAILY_LIMIT: int = 10
    MEDIUM_TIER_DAILY_LIMIT: int = 100
    PREMIUM_TIER_DAILY_LIMIT: int = 10000

    # ─── ML ───────────────────────────────────────────────
    MODEL_STORE_PATH: str = "./ml_models"
    MODEL_RETRAIN_INTERVAL_DAYS: int = 30

    # ─── Reports ──────────────────────────────────────────
    REPORT_STORAGE_PATH: str = "./reports/generated"
    REPORT_RETENTION_DAYS: int = 90

    # ─── Monitoring ───────────────────────────────────────
    PROMETHEUS_ENABLED: bool = True
    SENTRY_DSN: Optional[str] = None

    # ─── CORS ─────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080,http://localhost:5173"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    # ─── Logging ──────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # ─── Cameroon-specific ────────────────────────────────
    CAMEROON_REGIONS: List[str] = [
        "North West", "South West", "West", "Littoral",
        "Adamawa", "Far North", "Centre", "East", "North", "South"
    ]

    SUPPORTED_CROPS: List[str] = [
        "maize", "rice", "cassava", "cocoyam", "plantain",
        "cocoa", "coffee", "groundnut", "beans", "tomato",
        "onion", "potato", "palm_oil", "sorghum"
    ]

    SUBSCRIPTION_PLANS: dict = {
        "free": {
            "daily_limit": 10,
            "accuracy_tier": "basic",
            "report_formats": ["json"],
            "forecast_granularity": "monthly",
        },
        "medium": {
            "daily_limit": 100,
            "accuracy_tier": "standard",
            "report_formats": ["json", "pdf"],
            "forecast_granularity": "weekly",
        },
        "premium": {
            "daily_limit": 10000,
            "accuracy_tier": "advanced",
            "report_formats": ["json", "pdf"],
            "forecast_granularity": "daily",
        },
    }

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — loaded once at startup."""
    return Settings()


settings = get_settings()
