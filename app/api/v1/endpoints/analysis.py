"""
API Routes — ML Analysis Endpoints
=====================================
POST /price-forecasting/{crop}
POST /crop-suitability
POST /yield-prediction/{crop}
POST /risk-score

All endpoints:
1. Check Redis cache by request hash
2. Check PostgreSQL for existing result
3. If cache miss → run ML pipeline
4. Store result + update cache
5. Return result with cache flag
"""

import time
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.hashing import (
    hash_price_forecast,
    hash_crop_suitability,
    hash_yield_prediction,
    hash_risk_score,
)
from app.core.logging import api_logger
from app.core.security import get_current_user
from app.db.session import get_db
from app.ml.pipelines.price_forecasting import run_price_forecast
from app.ml.pipelines.crop_suitability import run_crop_suitability
from app.ml.pipelines.yield_and_risk import run_yield_prediction, run_risk_score
from app.models.models import (
    PriceForecast,
    SuitabilityResult,
    YieldPrediction,
    RiskScore,
    RequestLog,
    RequestStatus,
    User,
)
from app.schemas.schemas import (
    PriceForecastRequest,
    PriceForecastResponse,
    CropSuitabilityRequest,
    CropSuitabilityResponse,
    YieldPredictionRequest,
    YieldPredictionResponse,
    RiskScoreRequest,
    RiskScoreResponse,
)
from app.services.cache import cache_get, cache_set

router = APIRouter(tags=["ML Analysis"])


# ─── Helper: Cache-or-Compute ─────────────────────────────────────────────────


async def _check_db_cache(
    request_hash: str,
    model_class,
    db: AsyncSession,
):
    """Check PostgreSQL for existing result with matching hash."""
    result = await db.execute(
        select(model_class).where(model_class.request_hash == request_hash)
    )
    return result.scalar_one_or_none()


type_map = {
    "price_forecast": PriceForecast,
    "crop_suitability": SuitabilityResult,
    "yield_prediction": YieldPrediction,
    "risk_score": RiskScore,
}


async def _log_request(
    user_id,
    request_hash: str,
    endpoint: str,
    parameters: dict,
    is_cached: bool,
    db: AsyncSession,
) -> RequestLog:
    """Create a request log entry."""
    log = RequestLog(
        user_id=user_id,
        request_hash=request_hash,
        endpoint=endpoint,
        parameters=parameters,
        result_table=type_map.get(endpoint).__tablename__,
        status=RequestStatus.completed,
        is_cached=is_cached,
    )
    db.add(log)
    await db.flush()
    TargetModel = type_map.get(endpoint)
    if TargetModel is not None:
        # Build the update expression programmatically
        result = await db.execute(
            select(TargetModel.id)
            .where(TargetModel.request_hash == request_hash)
            .limit(1)
        )

        result_id = result.scalar_one_or_none()
        stmt = (
            update(RequestLog)
            .where(RequestLog.request_hash == request_hash)
            .where(RequestLog.endpoint == endpoint)
            .where(RequestLog.result_id == None)
            .values(
                result_id=result_id,
            )
        )

        result = await db.execute(stmt)
        await db.commit()
        api_logger.info(
            f"Backfilled request logs for endpoint '{endpoint}'",
            updated_rows=result.rowcount,
        )
    return log


# ─── Crop Suitability ─────────────────────────────────────────────────────────
def _flatten_suitability_response(data: dict) -> dict:
    """
    Normalise any suitability result dict to the flat structure
    expected by CropSuitabilityResponse, regardless of whether
    it came from Redis, DB, or a fresh ML computation.

    Handles both formats:
      - Nested:  data["environmental"]["rainfall_mm"]  (ML output)
      - Flat:    data["rainfall"]                      (DB / old cache)
    """
    env = data.get("environmental", {})

    return {
        "best_crops": data.get("best_crops", []),
        "suitability_scores": data.get("suitability_scores", {}),
        "recommended_crop": data.get("recommended_crop", ""),
        # Prefer flat keys (DB/old-cache), fall back to suffixed env keys (ML output)
        "soil_ph": data.get("soil_ph") or env.get("soil_ph"),
        "rainfall": data.get("rainfall") or env.get("rainfall_mm"),
        "temperature": data.get("temperature") or env.get("temperature_c"),
        "humidity": data.get("humidity") or env.get("humidity_pct"),
        "elevation": data.get("elevation") or env.get("elevation_m"),
        "region_estimate": data.get("region_estimate", "Cameroon"),
        "model_used": data.get("model_used", ""),
        "metrics": data.get("metrics") or data.get("accuracy_metrics", {}),
        "subscription_tier": data.get("subscription_tier", ""),
        "cached": data.get("cached", False),
        "generated_at": data.get("generated_at", ""),
    }


@router.post(
    "/crop-suitability",
    response_model=CropSuitabilityResponse,
    summary="Determine best crops for a location",
)
async def crop_suitability(
    payload: CropSuitabilityRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tier = current_user.subscription_type.value
    req_hash = hash_crop_suitability(payload.latitude, payload.longitude, tier)

    # ── Redis cache ───────────────────────────────────────────────
    cached = await cache_get(req_hash)
    if cached:
        cached["cached"] = True
        # ✅ Flatten before returning — Redis may hold old nested format
        return _flatten_suitability_response(cached)

    # ── DB cache ──────────────────────────────────────────────────
    db_result = await _check_db_cache(req_hash, SuitabilityResult, db)
    if db_result:
        response = _flatten_suitability_response(
            {
                "best_crops": db_result.best_crops,
                "suitability_scores": db_result.suitability_scores,
                "recommended_crop": db_result.recommended_crop,
                "soil_ph": db_result.soil_ph,
                "rainfall": db_result.rainfall,
                "temperature": db_result.temperature,
                "humidity": db_result.humidity,
                "elevation": db_result.elevation,
                "region_estimate": getattr(db_result, "region_estimate", "Cameroon"),
                "model_used": db_result.model_used,
                "metrics": db_result.accuracy_metrics,
                "subscription_tier": db_result.subscription_tier,
                "cached": True,
                "generated_at": db_result.created_at.isoformat(),
            }
        )
        await cache_set(req_hash, response, tier)
        await _log_request(
            current_user.id,
            req_hash,
            "crop_suitability",
            payload.model_dump(),
            True,
            db,
        )
        return response

    # ── Fresh ML computation ──────────────────────────────────────
    ml_result = await run_crop_suitability(
        payload.latitude,
        payload.longitude,
        tier,
    )
    env = ml_result["environmental"]

    # Persist to DB
    db.add(
        SuitabilityResult(
            request_hash=req_hash,
            latitude=payload.latitude,
            longitude=payload.longitude,
            soil_ph=env.get("soil_ph"),
            rainfall=env.get("rainfall_mm"),
            temperature=env.get("temperature_c"),
            humidity=env.get("humidity_pct"),
            elevation=env.get("elevation_m"),
            best_crops=ml_result["best_crops"],
            suitability_scores=ml_result["suitability_scores"],
            recommended_crop=ml_result["recommended_crop"],
            region_estimate=ml_result["region_estimate"],
            model_used=ml_result["model_used"],
            accuracy_metrics=ml_result["metrics"],
            subscription_tier=tier,
        )
    )
    await db.commit()
    await _log_request(
        current_user.id, req_hash, "crop_suitability", payload.model_dump(), False, db
    )

    # ✅ Flatten before caching and returning
    response = _flatten_suitability_response(ml_result)
    response["cached"] = False

    # Store the already-flat dict so future Redis hits never need re-flattening
    await cache_set(req_hash, response, tier)
    return response


# ─── Price Forecasting ────────────────────────────────────────────────────────
@router.post(
    "/price-forecasting/{crop}",
    response_model=PriceForecastResponse,
    summary="Forecast future crop prices",
    description="""
    Predicts future market prices for a specified Cameroon crop.

    **Algorithms by subscription tier:**
    - FREE: ARIMA (basic seasonality)
    - MEDIUM: Prophet (improved accuracy with dual seasonality)
    - PREMIUM: Ensemble ARIMA + Prophet (highest accuracy)

    **Deduplication:** Identical requests return cached results instantly.
    """,
)
async def price_forecasting(
    crop: str = Path(..., description="Crop name (e.g. maize, rice, cocoa)"),
    payload: PriceForecastRequest = ...,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    crop = crop.lower().replace("-", "_").replace(" ", "_")

    if crop not in settings.SUPPORTED_CROPS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported crop '{crop}'. Supported: {settings.SUPPORTED_CROPS}",
        )

    tier = current_user.subscription_type.value

    # 1. Generate request fingerprint
    req_hash = hash_price_forecast(crop, payload.duration, payload.duration_type, tier)

    # 2. Check Redis cache first (fast path)
    cached = await cache_get(req_hash)
    if cached:
        api_logger.info(
            "Price forecast cache hit (Redis)", crop=crop, hash=req_hash[:8]
        )
        cached["cached"] = True
        return cached

    # 3. Check PostgreSQL for existing result
    db_result = await _check_db_cache(req_hash, PriceForecast, db)
    if db_result:
        api_logger.info("Price forecast cache hit (DB)", crop=crop, hash=req_hash[:8])
        result = {
            "crop": db_result.crop,
            "forecast_period": db_result.forecast_period,
            "currency": "XAF (CFA Franc)",
            "predictions": db_result.predictions,
            "model_used": db_result.model_used,
            "accuracy": db_result.accuracy_metrics,
            "subscription_tier": db_result.subscription_tier,
            "cached": True,
            "generated_at": db_result.created_at.isoformat(),
        }
        # Re-populate Redis
        await cache_set(req_hash, result, tier)
        await _log_request(
            current_user.id, req_hash, "price_forecast", payload.model_dump(), True, db
        )
        return result

    # 4. Cache miss — compute forecast
    start = time.perf_counter()
    result = await run_price_forecast(
        crop, payload.duration, payload.duration_type, tier
    )
    duration_ms = int((time.perf_counter() - start) * 1000)

    # 5. Persist to PostgreSQL
    forecast_row = PriceForecast(
        request_hash=req_hash,
        crop=crop,
        forecast_period=f"{payload.duration} {payload.duration_type}",
        duration=payload.duration,
        duration_type=payload.duration_type,
        model_used=result["model_used"],
        subscription_tier=tier,
        predictions=result["predictions"],
        accuracy_metrics=result["accuracy"],
    )
    db.add(forecast_row)

    # 6. Log request
    await _log_request(
        current_user.id, req_hash, "price_forecast", payload.model_dump(), False, db
    )

    # 7. Cache result in Redis
    await cache_set(req_hash, result, tier)

    api_logger.info(
        "Price forecast computed",
        crop=crop,
        tier=tier,
        duration_ms=duration_ms,
        model=result["model_used"],
    )

    return result


# ─── Yield Prediction ─────────────────────────────────────────────────────────


@router.post(
    "/yield-prediction/{crop}",
    response_model=YieldPredictionResponse,
    summary="Predict crop yield",
)
async def yield_prediction(
    crop: str = Path(..., description="Crop name"),
    payload: YieldPredictionRequest = ...,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    crop = crop.lower().replace("-", "_").replace(" ", "_")

    if crop not in settings.SUPPORTED_CROPS:
        raise HTTPException(422, f"Unsupported crop '{crop}'")

    tier = current_user.subscription_type.value
    fertilizer = payload.fertilizer_type or "none"

    req_hash = hash_yield_prediction(
        crop,
        payload.land_size,
        payload.latitude,
        payload.longitude,
        payload.irrigation,
        fertilizer,
        tier,
    )

    cached = await cache_get(req_hash)
    if cached:
        cached["cached"] = True
        return cached

    db_result = await _check_db_cache(req_hash, YieldPrediction, db)
    if db_result:
        result = {
            "crop": crop,
            "predicted_yield": f"{db_result.yield_per_hectare} tons/hectare",
            "total_yield_tons": db_result.predicted_yield_tons,
            "yield_per_hectare": db_result.yield_per_hectare,
            "confidence": db_result.confidence,
            "model_used": db_result.model_used,
            "metrics": db_result.accuracy_metrics,
            "subscription_tier": db_result.subscription_tier,
            "cached": True,
            "generated_at": db_result.created_at.isoformat(),
        }
        await cache_set(req_hash, result, tier)
        await _log_request(
            current_user.id,
            req_hash,
            "yield_prediction",
            payload.model_dump(),
            True,
            db,
        )
        return result

    result = await run_yield_prediction(
        crop,
        payload.land_size,
        payload.latitude,
        payload.longitude,
        payload.irrigation,
        fertilizer,
        tier,
    )

    env = result["environmental"]

    db.add(
        YieldPrediction(
            request_hash=req_hash,
            crop=crop,
            land_size=payload.land_size,
            soil_ph=env.get("soil_ph"),
            rainfall=env.get("rainfall_mm"),
            temperature=env.get("temperature_c"),
            humidity=env.get("humidity_pct"),
            elevation=env.get("elevation_m"),
            irrigation=payload.irrigation,
            fertilizer_type=fertilizer,
            predicted_yield_tons=result["total_yield_tons"],
            yield_per_hectare=result["yield_per_hectare"],
            confidence=result["confidence"],
            model_used=result["model_used"],
            accuracy_metrics=result["metrics"],
            subscription_tier=tier,
        )
    )
    await _log_request(
        current_user.id, req_hash, "yield_prediction", payload.model_dump(), False, db
    )
    await cache_set(req_hash, result, tier)

    return result


# ─── Risk Score ───────────────────────────────────────────────────────────────


@router.post(
    "/risk-score",
    response_model=RiskScoreResponse,
    summary="Generate agricultural risk score",
)
async def risk_score(
    payload: RiskScoreRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    tier = current_user.subscription_type.value

    req_hash = hash_risk_score(
        payload.crop,
        payload.land_size,
        payload.latitude,
        payload.longitude,
        payload.market_access,
        tier,
    )

    cached = await cache_get(req_hash)
    if cached:
        cached["cached"] = True
        return cached

    db_result = await _check_db_cache(req_hash, RiskScore, db)
    if db_result:
        result = {
            "overall_risk_score": db_result.overall_risk_score,
            "risk_level": db_result.risk_level.value,
            "financial_risk": db_result.financial_risk,
            "climate_risk": db_result.climate_risk,
            "agronomic_risk": db_result.agronomic_risk,
            "risk_factors": db_result.risk_factors,
            "recommendations": db_result.recommendations,
            "metrics": db_result.accuracy_metrics,
            "subscription_tier": db_result.subscription_tier,
            "cached": True,
            "generated_at": db_result.created_at.isoformat(),
        }

        await cache_set(req_hash, result, tier)
        await _log_request(
            current_user.id, req_hash, "risk_score", payload.model_dump(), True, db
        )
        return result

    result = await run_risk_score(
        payload.crop,
        payload.land_size,
        payload.latitude,
        payload.longitude,
        payload.market_access,
        tier,
    )

    from app.models.models import RiskLevel

    risk_level_enum = RiskLevel(result["risk_level"])

    env = result["environmental"]

    db.add(
        RiskScore(
            request_hash=req_hash,
            crop=payload.crop,
            region=result["region"],
            land_size=payload.land_size,
            soil_ph=env.get("soil_ph"),
            rainfall=env.get("rainfall_mm"),
            temperature=env.get("temperature_c"),
            market_access=payload.market_access,
            overall_risk_score=result["overall_risk_score"],
            risk_level=risk_level_enum,
            financial_risk=result["financial_risk"],
            climate_risk=result["climate_risk"],
            agronomic_risk=result["agronomic_risk"],
            risk_factors=result["risk_factors"],
            recommendations=result["recommendations"],
            accuracy_metrics=result["metrics"],
            subscription_tier=tier,
        )
    )
    await _log_request(
        current_user.id, req_hash, "risk_score", payload.model_dump(), False, db
    )
    await cache_set(req_hash, result, tier)

    return result
