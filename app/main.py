"""
GEoN Risk Assessment Platform — Main Application
=====================================================
FastAPI application factory with:
  - Lifespan management (startup/shutdown)
  - Middleware registration
  - API router mounting
  - Prometheus metrics
  - OpenAPI documentation customization
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.api.v1.router import api_router
from app.middleware.middleware import register_middleware
from app.services.cache import close_redis


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Runs setup on startup, teardown on shutdown.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    setup_logging()
    logger.info(
        "Starting GEoN Platform",
        version=settings.APP_VERSION,
        env=settings.APP_ENV,
    )

    # Ensure required directories exist
    for path in [settings.MODEL_STORE_PATH, settings.REPORT_STORAGE_PATH,
                 "data/prices", "data/yields", "data/soil", "data/climate",
                 "ml_models/price", "ml_models/suitability",
                 "ml_models/yield", "ml_models/risk"]:
        os.makedirs(path, exist_ok=True)

    # Check if datasets exist; generate if missing
    prices_dir = "data/prices"
    if not os.path.isdir(prices_dir) or len(os.listdir(prices_dir)) == 0:
        logger.warning(
            "No price datasets found. Run: python scripts/generate_datasets.py"
        )

    # Prometheus metrics
    if settings.PROMETHEUS_ENABLED:
        try:
            from prometheus_fastapi_instrumentator import Instrumentator
            Instrumentator(
                should_group_status_codes=True,
                should_ignore_untemplated=True,
                should_respect_env_var=True,
                should_instrument_requests_inprogress=True,
                excluded_handlers=["/health", "/metrics"],
            ).instrument(app).expose(app, endpoint="/metrics")
            logger.info("Prometheus metrics enabled at /metrics")
        except ImportError:
            logger.warning("prometheus_fastapi_instrumentator not installed")

    # Sentry (production error tracking)
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            sentry_sdk.init(dsn=settings.SENTRY_DSN, environment=settings.APP_ENV)
            logger.info("Sentry error tracking initialized")
        except ImportError:
            pass

    logger.info("✅ Application startup complete")

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down GEoN Platform...")
    await close_redis()
    logger.info("✅ Graceful shutdown complete")


# ─── Application Factory ──────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
## 🌱 GEoN Agricultural Risk Assessment Platform

AI-powered agricultural analytics for Cameroon farmers, agribusinesses, and cooperatives.

### Features
- **Price Forecasting** — ARIMA / Prophet / Ensemble predictions for 14 crops
- **Crop Suitability** — ML-based crop recommendations by soil & climate
- **Yield Prediction** — Regression models for harvest estimation
- **Risk Scoring** — Financial, climate & agronomic risk assessment
- **Reports** — Downloadable PDF and JSON reports

### Subscription Tiers
| Feature | Free | Medium | Premium |
|---------|------|--------|---------|
| Daily Requests | 10 | 100 | Unlimited |
| Model Quality | Basic | Standard | Ensemble |
| Report Formats | JSON | JSON+PDF | JSON+PDF |
| Forecast Granularity | Monthly | Weekly | Daily |

### Authentication
All analysis endpoints require a Bearer JWT token.
Register at `/api/v1/auth/register`, then login at `/api/v1/auth/login`.

### Request Deduplication
Identical requests (same parameters + subscription tier) return cached results
immediately — no recomputation cost.
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "GEoN Support",
            "email": "support@GEoN.cm",
        },
        license_info={
            "name": "MIT",
        },
    )

    # Register middleware (order matters — see middleware.py)
    register_middleware(app)

    # Mount API routes
    app.include_router(api_router)

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# ─── App Instance ─────────────────────────────────────────────────────────────
app = create_app()


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT or 8000,
        workers=1,          # Use 1 for development; set via env for production
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=False,   # Handled by RequestLoggingMiddleware
    )
