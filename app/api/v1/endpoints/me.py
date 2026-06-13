"""
History Endpoint
==================
GET  /history/my              — List user's history of all analyses (forecasts, suitability, yield predictions, risk scores) with metadata and timestamps, sorted by creation date.

"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.session import get_db
from app.models.models import (
    PriceForecast, RequestLog, RequestStatus, SuitabilityResult,
    User, YieldPrediction, RiskScore
)
from app.schemas.schemas import HistoryResponse
from app.core.logging import api_logger

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/my", response_model=HistoryResponse, status_code=status.HTTP_200_OK, summary="Get complete user analysis history")
async def get_user_all_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetches every completed analysis calculation ever requested by the user
    by resolving records through the Request Deduplication Log.
    """
    # 1. Fetch all completed requests for this user, ordered newest first
    logs_stmt = (
        select(RequestLog)
        .where(
            RequestLog.user_id == current_user.id,
            RequestLog.status == RequestStatus.completed,
            RequestLog.result_id.isnot(None)
        )
        .order_by(RequestLog.created_at.desc())
    )
    logs_result = await db.execute(logs_stmt)
    request_logs = logs_result.scalars().all()

    if not request_logs:
        return {"user_id": current_user.id, "total_activities": 0, "history": []}

    # 2. Group result IDs by their target tables to avoid N+1 query loops
    table_map = {
        "forecasts": [],
        "suitability_results": [],
        "yield_predictions": [],
        "risk_scores": []
    }
    for log in request_logs:
        if log.result_table in table_map:
            table_map[log.result_table].append(log.result_id)

    # 3. Fetch specific records asynchronously in parallel batches
    tasks = []
    if table_map["forecasts"]:
        tasks.append(db.execute(select(PriceForecast).where(PriceForecast.id.in_(table_map["forecasts"]))))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    if table_map["suitability_results"]:
        tasks.append(db.execute(select(SuitabilityResult).where(SuitabilityResult.id.in_(table_map["suitability_results"]))))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    if table_map["yield_predictions"]:
        tasks.append(db.execute(select(YieldPrediction).where(YieldPrediction.id.in_(table_map["yield_predictions"]))))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    if table_map["risk_scores"]:
        tasks.append(db.execute(select(RiskScore).where(RiskScore.id.in_(table_map["risk_scores"]))))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    # Wait for all batches to return
    f_res, s_res, y_res, r_res = await asyncio.gather(*tasks)

    # 4. Map DB entities into standard lookup dictionaries by their primary key UUIDs
    data_lookups = {
        "forecasts": {row.id: row for row in f_res.scalars().all()} if f_res else {},
        "suitability_results": {row.id: row for row in s_res.scalars().all()} if s_res else {},
        "yield_predictions": {row.id: row for row in y_res.scalars().all()} if y_res else {},
        "risk_scores": {row.id: row for row in r_res.scalars().all()} if r_res else {},
    }

    # 5. Compile the timeline using original request logs to preserve sequence order
    history_timeline = []

    for log in request_logs:
        table = log.result_table
        r_id = log.result_id
        # Pull corresponding record from dictionary maps
        record = data_lookups.get(table, {}).get(r_id)
        if not record:
            continue  # Fallback safety check if target data was dropped

        payload = {
            "request_id": log.id,
            "result_id": record.id,
            "analysis_type": table,
            "is_cached_hit": log.is_cached,
            "created_at": log.created_at,
        }

        # Handle unique column mappings explicitly using your exact database fields
        if table == "forecasts":
            payload.update({
                "crop": record.crop.title(),
                "summary": f"Price forecast for {record.forecast_period} ({record.duration} {record.duration_type}s)",
                "meta": {"model": record.model_used, "tier": record.subscription_tier}
            })
            
        elif table == "suitability_results":
            payload.update({
                "crop": record.recommended_crop.title(),
                "summary": f"Crop suitability match for region near {record.region_estimate}",
                "meta": {"coordinates": [record.latitude, record.longitude], "ph": record.soil_ph}
            })
            
        elif table == "yield_predictions":
            payload.update({
                "crop": record.crop.title(),
                "summary": f"Predicted yield of {record.predicted_yield_tons:,.1f} tons over {record.land_size} ha",
                "meta": {"yield_per_hectare": record.yield_per_hectare, "confidence": record.confidence}
            })
            
        elif table == "risk_scores":
            payload.update({
                "crop": record.crop.title(),
                "summary": f"Risk rating evaluated as {record.risk_level.value.upper()} ({record.overall_risk_score:.2f})",
                "meta": {"region": record.region, "climate_risk": record.climate_risk, "financial_risk": record.financial_risk}
            })

        history_timeline.append(payload)

    return {
        "user_id": current_user.id,
        "total_activities": len(history_timeline),
        "history": history_timeline
    }