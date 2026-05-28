"""
Health Check & Monitoring Endpoints
=====================================
GET /health          — Basic health check
GET /health/detailed — Full system status (DB, Redis, ML models)
GET /metrics         — Prometheus metrics (handled by instrumentator)
"""

import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.services.cache import ping_redis
from app.schemas.schemas import HealthResponse

router = APIRouter(tags=["Health & Monitoring"])


@router.get("/health", response_model=HealthResponse, summary="Basic health check")
async def health_check():
    """Lightweight liveness probe — returns immediately."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        services={"api": "up"},
    )


@router.get("/health/detailed", summary="Full system health check")
async def detailed_health(db: AsyncSession = Depends(get_db)):
    """
    Checks all dependent services:
    - PostgreSQL connectivity
    - Redis connectivity
    - ML model availability
    - Dataset availability
    """
    services = {}

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    try:
        await db.execute(text("SELECT 1"))
        services["postgresql"] = "up"
    except Exception as e:
        services["postgresql"] = f"down: {str(e)[:80]}"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_ok = await ping_redis()
    services["redis"] = "up" if redis_ok else "down"

    # ── ML Models ─────────────────────────────────────────────────────────────
    model_root = settings.MODEL_STORE_PATH
    ml_dirs = ["price", "suitability", "yield", "risk"]
    ml_status = {}
    for d in ml_dirs:
        path = os.path.join(model_root, d)
        if os.path.isdir(path):
            model_files = [f for f in os.listdir(path) if f.endswith(".pkl")]
            ml_status[d] = f"{len(model_files)} models cached"
        else:
            ml_status[d] = "not trained (will train on first request)"
    services["ml_models"] = ml_status

    # ── Datasets ──────────────────────────────────────────────────────────────
    prices_dir = os.path.join("data", "prices")
    if os.path.isdir(prices_dir):
        csv_count = len([f for f in os.listdir(prices_dir) if f.endswith(".csv")])
        services["datasets"] = f"{csv_count}/14 crop price datasets available"
    else:
        services["datasets"] = "missing (run: python scripts/generate_datasets.py)"

    overall = "healthy" if all(
        v == "up" or isinstance(v, dict) or "cached" in str(v) or "available" in str(v)
        for k, v in services.items()
        if k in ["postgresql", "redis"]
    ) else "degraded"

    return {
        "status": overall,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services,
        "supported_crops": settings.SUPPORTED_CROPS,
        "supported_regions": settings.CAMEROON_REGIONS,
    }


@router.get("/health/ready", summary="Readiness probe for Kubernetes")
async def readiness_probe(db: AsyncSession = Depends(get_db)):
    """K8s readiness probe — checks DB and Redis are reachable."""
    try:
        await db.execute(text("SELECT 1"))
        redis_ok = await ping_redis()
        if not redis_ok:
            return {"status": "not ready", "reason": "Redis unreachable"}, 503
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not ready", "reason": str(e)}, 503
