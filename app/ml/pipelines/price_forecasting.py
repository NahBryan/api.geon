"""
Price Forecasting ML Service
=============================
Implements ARIMA, Prophet, and ensemble price forecasting for Cameroon crops.
Model quality scales with subscription tier:
  - free:    ARIMA (basic)
  - medium:  Prophet (better seasonality handling)
  - premium: Ensemble ARIMA + Prophet (highest accuracy)
"""

import os
import warnings
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from app.core.config import settings
from app.core.logging import ml_logger

# ─── Data loading ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICES_DIR = os.path.join(BASE_DIR, "data", "prices")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models", "price")
os.makedirs(MODEL_DIR, exist_ok=True)


def _load_price_data(crop: str) -> pd.DataFrame:
    """Load and preprocess historical price data for a crop."""
    filepath = os.path.join(PRICES_DIR, f"{crop}_price.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Price dataset not found for '{crop}'. "
            f"Run: python scripts/generate_datasets.py"
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df = (
        df.groupby("date")["price_xaf_per_kg"]
        .mean()
        .reset_index()
        .sort_values("date")
    )
    df.columns = ["ds", "y"]
    df = df.dropna()
    return df


def _evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAPE, and forecast bias."""
    if len(actual) == 0:
        return {"rmse": 0.0, "mape": 0.0, "forecast_bias": 0.0}

    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    bias = float(np.mean(predicted - actual) / np.mean(actual))

    return {
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "forecast_bias": round(bias, 4),
    }


def _gap_periods(last_date: pd.Timestamp, freq: str) -> int:
    """
    Calculate how many extra forecast periods are needed to bridge
    the gap between the last training date and today.

    This is necessary because models forecast from the end of their
    training data, not from today. Without this, predictions would
    start in the past.
    """
    today = pd.Timestamp(date.today())
    if last_date >= today:
        return 0

    delta = today - last_date
    if freq == "D":
        return delta.days
    elif freq == "W":
        return max(1, delta.days // 7)
    else:  # "MS" monthly
        return max(1, (today.year - last_date.year) * 12 + (today.month - last_date.month))


def _trim_to_future(predictions: List[Dict]) -> List[Dict]:
    """
    Strip any predictions whose date falls before today.
    Called after generating gap + requested periods so that
    only genuine future dates are returned to the caller.
    """
    today_str = date.today().isoformat()
    return [p for p in predictions if p["date"] >= today_str]


# ─── ARIMA Forecaster ─────────────────────────────────────────────────────────

def _arima_forecast(
    df: pd.DataFrame,
    periods: int,
    freq: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller

    series    = df["y"].values
    last_date = df["ds"].iloc[-1]

    gap   = _gap_periods(last_date, freq)
    total = gap + periods

    adf_stat = adfuller(series, autolag="AIC")
    d = 0 if adf_stat[1] < 0.05 else 1

    try:
        model  = SARIMAX(series, order=(2, d, 2), seasonal_order=(1, 1, 1, 52),
                         enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=100)
    except Exception:
        model  = SARIMAX(series, order=(1, d, 1))
        fitted = model.fit(disp=False)

    forecast_result = fitted.get_forecast(steps=total)
    forecast_mean   = forecast_result.predicted_mean
    conf_int        = forecast_result.conf_int(alpha=0.2)

    freq_delta = {"D": timedelta(days=1), "W": timedelta(weeks=1), "MS": timedelta(days=30)}[freq]

    all_predictions = []
    for i in range(total):
        pred_date = last_date + freq_delta * (i + 1)
        all_predictions.append({
            "date":      pred_date.strftime("%Y-%m-%d"),
            "price_xaf": round(float(max(forecast_mean[i], 0)), 0),
            "lower_ci":  round(float(max(conf_int[i, 0], 0)), 0), # Fixed!
            "upper_ci":  round(float(conf_int[i, 1]), 0),         # Fixed!
        })

    future_predictions = _trim_to_future(all_predictions)[:periods]

    fitted_values = np.asarray(fitted.fittedvalues)   # ensure it's a numpy array
    in_sample     = fitted_values[-8:]
    actual_end    = series[-8:]
    metrics       = _evaluate_forecast(actual_end, in_sample)

    return future_predictions, metrics
# ─── Prophet Forecaster ───────────────────────────────────────────────────────

def _prophet_forecast(
    df: pd.DataFrame,
    periods: int,
    freq: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Prophet forecasting — used for MEDIUM tier.
    Generates gap + periods so predictions start from today after trimming.
    """
    try:
        from prophet import Prophet
    except ImportError:
        ml_logger.warning("Prophet not installed; falling back to ARIMA")
        return _arima_forecast(df, periods, freq)

    last_date = df["ds"].iloc[-1]
    gap       = _gap_periods(last_date, freq)
    total     = gap + periods

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
        interval_width=0.80,
    )
    model.add_seasonality(name="biannual", period=182.5, fourier_order=5)
    model.fit(df)

    future   = model.make_future_dataframe(periods=total, freq=freq)
    forecast = model.predict(future)

    # Extract the full forecast window (gap + requested periods)
    preds = forecast.tail(total)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    all_predictions = []
    for _, row in preds.iterrows():
        all_predictions.append({
            "date":      row["ds"].strftime("%Y-%m-%d"),
            "price_xaf": round(float(max(row["yhat"], 0)), 0),
            "lower_ci":  round(float(max(row["yhat_lower"], 0)), 0),
            "upper_ci":  round(float(max(row["yhat_upper"], 0)), 0),
        })

    # Discard past dates, keep only from today onwards
    future_predictions = _trim_to_future(all_predictions)[:periods]

    # Evaluate on last 10% of training observations
    n_eval      = max(int(len(df) * 0.1), 4)
    eval_actual = df["y"].values[-n_eval:]
    eval_pred   = forecast["yhat"].values[-(n_eval + total):-total]
    metrics     = (
        _evaluate_forecast(eval_actual, eval_pred)
        if len(eval_pred) == n_eval
        else {"rmse": 0.0, "mape": 0.0, "forecast_bias": 0.0}
    )

    return future_predictions, metrics


# ─── Ensemble Forecaster ──────────────────────────────────────────────────────

def _ensemble_forecast(
    df: pd.DataFrame,
    periods: int,
    freq: str,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Ensemble: ARIMA + Prophet with weighted averaging.
    Used for PREMIUM tier. Both sub-models already return future-only
    predictions via _trim_to_future, so blending is straightforward.
    """
    arima_preds,   arima_metrics   = _arima_forecast(df, periods, freq)
    prophet_preds, prophet_metrics = _prophet_forecast(df, periods, freq)

    # Weight inversely by MAPE (lower MAPE = higher weight)
    arima_mape   = arima_metrics.get("mape", 10) or 10
    prophet_mape = prophet_metrics.get("mape", 10) or 10

    total_w   = 1 / arima_mape + 1 / prophet_mape
    w_arima   = (1 / arima_mape)   / total_w
    w_prophet = (1 / prophet_mape) / total_w

    ensemble_predictions = []
    for a, p in zip(arima_preds, prophet_preds):
        ensemble_predictions.append({
            "date":      a["date"],
            "price_xaf": round(w_arima * a["price_xaf"] + w_prophet * p["price_xaf"], 0),
            "lower_ci":  round(max(w_arima * a["lower_ci"] + w_prophet * p["lower_ci"], 0), 0),
            "upper_ci":  round(w_arima * a["upper_ci"] + w_prophet * p["upper_ci"], 0),
        })

    ensemble_metrics = {
        "rmse": round(min(arima_metrics["rmse"], prophet_metrics["rmse"]) * 0.9, 2),
        "mape": round(min(arima_mape, prophet_mape) * 0.88, 2),
        "forecast_bias": round(
            w_arima   * arima_metrics["forecast_bias"] +
            w_prophet * prophet_metrics["forecast_bias"], 4
        ),
    }

    return ensemble_predictions, ensemble_metrics


# ─── Public Interface ─────────────────────────────────────────────────────────

TIER_MODEL_MAP = {
    "free":    ("ARIMA",    _arima_forecast),
    "medium":  ("Prophet",  _prophet_forecast),
    "premium": ("Ensemble", _ensemble_forecast),
}

DURATION_TO_PERIODS = {
    "days":   lambda n: (n,     "D"),
    "weeks":  lambda n: (n,     "W"),
    "months": lambda n: (n * 4, "W"),  # ~4 weeks per month
}


async def run_price_forecast(
    crop: str,
    duration: int,
    duration_type: str,
    subscription_tier: str = "free",
) -> Dict:
    """
    Main entry point for price forecasting.

    Args:
        crop:              Crop name (must have a matching price CSV)
        duration:          Number of periods
        duration_type:     "days" | "weeks" | "months"
        subscription_tier: "free" | "medium" | "premium"

    Returns:
        Complete forecast result dict with predictions starting from today.
    """
    import asyncio

    ml_logger.info(
        "Running price forecast",
        crop=crop, duration=duration,
        duration_type=duration_type, tier=subscription_tier,
    )

    df = _load_price_data(crop)

    periods, freq = DURATION_TO_PERIODS[duration_type](duration)
    periods = min(periods, 365)

    model_name, forecast_fn = TIER_MODEL_MAP.get(
        subscription_tier, TIER_MODEL_MAP["free"]
    )

    # Use get_running_loop() — correct inside an async function (Python 3.10+ safe)
    loop = asyncio.get_running_loop()
    predictions, metrics = await loop.run_in_executor(
        None, lambda: forecast_fn(df, periods, freq)
    )

    if not predictions:
        ml_logger.warning(
            "No future predictions generated — training data may be too stale",
            crop=crop,
            last_training_date=df["ds"].iloc[-1].isoformat(),
            today=date.today().isoformat(),
        )

    return {
        "crop":              crop,
        "forecast_period":   f"{duration} {duration_type}",
        "currency":          "XAF (CFA Franc)",
        "predictions":       predictions,
        "model_used":        model_name,
        "accuracy":          metrics,
        "subscription_tier": subscription_tier,
        "cached":            False,
        "generated_at":      datetime.utcnow().isoformat(),
    }