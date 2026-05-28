"""
Celery Background Task Workers
================================
Handles CPU-intensive ML tasks asynchronously:
  - Model retraining
  - Batch report generation
  - Dataset refresh
  - Cache warming

Workers are separate processes that consume from Redis broker queues.
"""

import asyncio
import os
from datetime import datetime

from celery import Celery
from celery.schedules import crontab

from app.core.config import settings
from app.core.logging import task_logger

# ─── Celery App ───────────────────────────────────────────────────────────────

celery_app = Celery(
    "agri_risk_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="Africa/Douala",
    enable_utc=True,

    # Task routing — separate queues by priority
    task_routes={
        "app.workers.tasks.retrain_all_models": {"queue": "maintenance"},
        "app.workers.tasks.warm_cache": {"queue": "maintenance"},
        "app.workers.tasks.generate_report_async": {"queue": "reports"},
        "app.workers.tasks.refresh_datasets": {"queue": "maintenance"},
    },

    # Retry configuration
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Result expiry
    result_expires=3600,

    # Scheduled tasks (Celery Beat)
    beat_schedule={
        "retrain-models-weekly": {
            "task": "app.workers.tasks.retrain_all_models",
            "schedule": crontab(hour=2, minute=0, day_of_week=0),  # Sunday 2am
        },
        "warm-cache-daily": {
            "task": "app.workers.tasks.warm_cache",
            "schedule": crontab(hour=6, minute=0),  # Daily 6am
        },
    },
)


# ─── Tasks ────────────────────────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    name="app.workers.tasks.retrain_all_models",
)
def retrain_all_models(self):
    """
    Retrain all ML models with latest data.
    Runs weekly via Celery Beat.
    """
    task_logger.info("Starting model retraining", task_id=self.request.id)

    try:
        from app.core.config import settings

        crops = settings.SUPPORTED_CROPS
        tiers = ["free", "medium", "premium"]
        results = {"yield": {}, "suitability": {}}

        for tier in tiers:
            # Retrain suitability model
            try:
                from app.ml.pipelines.crop_suitability import _train_model
                import joblib

                pipeline, le, metrics = _train_model(tier)
                model_dir = os.path.join("ml_models", "suitability")
                os.makedirs(model_dir, exist_ok=True)
                joblib.dump(pipeline, os.path.join(model_dir, f"suitability_{tier}.pkl"))
                joblib.dump(le, os.path.join(model_dir, f"suitability_{tier}_le.pkl"))
                joblib.dump(metrics, os.path.join(model_dir, f"suitability_{tier}_metrics.pkl"))
                results["suitability"][tier] = metrics
                task_logger.info("Suitability model retrained", tier=tier, metrics=metrics)
            except Exception as e:
                task_logger.error("Suitability retraining failed", tier=tier, error=str(e))

            # Retrain yield models
            for crop in crops[:5]:  # Limit in background to avoid timeout
                try:
                    from app.ml.pipelines.yield_and_risk import _train_yield_model
                    import joblib

                    pipe, metrics = _train_yield_model(crop, tier)
                    model_dir = os.path.join("ml_models", "yield")
                    os.makedirs(model_dir, exist_ok=True)
                    joblib.dump(pipe, os.path.join(model_dir, f"yield_{crop}_{tier}.pkl"))
                    joblib.dump(metrics, os.path.join(model_dir, f"yield_{crop}_{tier}_metrics.pkl"))
                    results["yield"][f"{crop}_{tier}"] = metrics
                except Exception as e:
                    task_logger.error("Yield model retraining failed",
                                      crop=crop, tier=tier, error=str(e))

        task_logger.info("Model retraining completed", results_count=len(results))
        return {"status": "completed", "timestamp": datetime.utcnow().isoformat()}

    except Exception as exc:
        task_logger.error("Retraining task failed", error=str(exc))
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.generate_report_async",
)
def generate_report_async(self, report_id: str, analysis_type: str, data: dict, user_id: str):
    """
    Generate a report file asynchronously.
    Updates the report record in PostgreSQL when complete.
    """
    task_logger.info("Generating report", report_id=report_id, type=analysis_type)

    try:
        import json
        output_path = os.path.join(
            settings.REPORT_STORAGE_PATH,
            f"{analysis_type}_{report_id}.json"
        )
        os.makedirs(settings.REPORT_STORAGE_PATH, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, default=str, indent=2)

        task_logger.info("Report generated", report_id=report_id, path=output_path)
        return {"status": "ready", "path": output_path}

    except Exception as exc:
        task_logger.error("Report generation failed", report_id=report_id, error=str(exc))
        raise


@celery_app.task(name="app.workers.tasks.warm_cache")
def warm_cache():
    """
    Pre-warm Redis cache for common crop/region combinations.
    Runs daily at 6am Africa/Douala time.
    """
    task_logger.info("Starting cache warming")

    common_forecasts = [
        ("maize", 1, "months"),
        ("rice", 1, "months"),
        ("cassava", 1, "months"),
        ("plantain", 2, "months"),
        ("tomato", 1, "months"),
    ]

    warmed = 0
    for crop, duration, duration_type in common_forecasts:
        try:
            # Run sync version of forecast for cache warming
            import asyncio
            from app.ml.pipelines.price_forecasting import run_price_forecast

            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                run_price_forecast(crop, duration, duration_type, "free")
            )
            loop.close()

            from app.core.hashing import hash_price_forecast
            from app.services.cache import cache_set

            req_hash = hash_price_forecast(crop, duration, duration_type, "free")
            loop2 = asyncio.new_event_loop()
            loop2.run_until_complete(cache_set(req_hash, result, "free"))
            loop2.close()

            warmed += 1
            task_logger.info("Cache warmed", crop=crop)
        except Exception as e:
            task_logger.warning("Cache warming failed for crop", crop=crop, error=str(e))

    task_logger.info("Cache warming complete", warmed=warmed)
    return {"warmed": warmed}


@celery_app.task(name="app.workers.tasks.refresh_datasets")
def refresh_datasets():
    """
    Regenerate datasets from latest data.
    Can be triggered manually or scheduled.
    """
    task_logger.info("Refreshing datasets")
    try:
        import subprocess
        subprocess.run(
            ["python", "scripts/generate_datasets.py"],
            check=True, capture_output=True
        )
        task_logger.info("Datasets refreshed successfully")
        return {"status": "completed"}
    except Exception as e:
        task_logger.error("Dataset refresh failed", error=str(e))
        raise
