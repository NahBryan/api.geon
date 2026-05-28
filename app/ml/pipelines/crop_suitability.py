"""
Crop Suitability Analysis ML Pipeline
======================================
Predicts the best crops for a given location based on:
  - GPS coordinates
  - Soil pH
  - Rainfall
  - Temperature
  - Humidity
  - Elevation

Algorithms by tier:
  - free:    Decision Tree
  - medium:  Random Forest
  - premium: Gradient Boosted Ensemble (RF + XGBoost)
"""

import os
import warnings
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import httpx
import asyncio
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

from app.core.config import settings
from app.core.logging import ml_logger
from re import match

import geopandas as gpd
from shapely.geometry import Point
import os
# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models", "suitability")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SHAPEFILE_PATH = os.path.join(BASE_DIR, "data", "gadm41_CMR_shp", "gadm41_CMR_1.shp")
YIELDS_DIR = os.path.join(BASE_DIR, "data", "yields")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Model config ─────────────────────────────────────────────────────────────
FEATURES = ["soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "elevation_m"]
TARGET   = "crop"

# ─── GADM Shapefile (primary region resolver) ─────────────────────────────────
try:
    if os.path.exists(SHAPEFILE_PATH):
        CMR_REGIONS_GDF = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        ml_logger.info("Successfully loaded GADM Cameroon shapefile boundaries.")
    else:
        CMR_REGIONS_GDF = None
        ml_logger.warning(
            f"GADM Shapefile not found at {SHAPEFILE_PATH}. "
            "Falling back to bounding-box region lookup."
        )
except Exception as e:
    CMR_REGIONS_GDF = None
    ml_logger.error(f"Failed to load GADM Shapefile: {e}")

# Applied when live weather APIs fail — more accurate than one global fallback.
REGION_ENV_DEFAULTS = {
    "Far North":  {"rainfall_mm": 600,  "temperature_c": 35, "humidity_pct": 30, "elevation_m": 350},
    "North":      {"rainfall_mm": 900,  "temperature_c": 32, "humidity_pct": 45, "elevation_m": 400},
    "Adamawa":    {"rainfall_mm": 1400, "temperature_c": 22, "humidity_pct": 70, "elevation_m": 1100},
    "Littoral":   {"rainfall_mm": 3500, "temperature_c": 27, "humidity_pct": 90, "elevation_m": 50},
    "South West": {"rainfall_mm": 2500, "temperature_c": 23, "humidity_pct": 85, "elevation_m": 900},
    "North West": {"rainfall_mm": 2200, "temperature_c": 19, "humidity_pct": 82, "elevation_m": 1500},
    "West":       {"rainfall_mm": 1800, "temperature_c": 20, "humidity_pct": 80, "elevation_m": 1400},
    "Centre":     {"rainfall_mm": 1600, "temperature_c": 24, "humidity_pct": 78, "elevation_m": 700},
    "East":       {"rainfall_mm": 1700, "temperature_c": 25, "humidity_pct": 80, "elevation_m": 650},
    "South":      {"rainfall_mm": 2000, "temperature_c": 25, "humidity_pct": 88, "elevation_m": 500},
}

# ─── Region resolution ────────────────────────────────────────────────────────
def _coords_to_region(lat: float, lon: float, path=SHAPEFILE_PATH) -> str:
    """
    Resolve GPS coordinates to a Cameroon administrative region.
    Primary:  GADM shapefile polygon lookup (precise).
    Fallback: Bounding-box scan (approximate, no file dependency).
    """
    if not os.path.exists(path):
        return "Center"
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    detected = Point(lon, lat)  # shapely is (lon, lat)
    match = gdf[gdf.geometry.contains(detected)]
    if not match.empty:
        return match.iloc[0]["NAME_1"]
    return detected, "Center"

# ─── Training data ────────────────────────────────────────────────────────────

def _get_training_data() -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load real yield dataset if present, otherwise generate synthetic data."""
    filepath = os.path.join(YIELDS_DIR, "cameroon_yields.csv")
    if not os.path.exists(filepath):
        ml_logger.warning("Real yield dataset not found — using synthetic training data.")
        return _generate_synthetic_training_data()

    df = pd.read_csv(filepath)
    X  = df[FEATURES].values
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET].values)
    return X, y, le


def _generate_synthetic_training_data() -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Generate synthetic training data based on known Cameroon crop agro-climatic ranges."""
    np.random.seed(42)
    records = []

    crop_profiles = {
        "maize":     {"ph": (5.5, 7.5), "rain": (600,  1200), "temp": (18, 32), "hum": (50, 85), "elev": (0,   2000)},
        "rice":      {"ph": (5.0, 7.0), "rain": (1200, 2500), "temp": (22, 35), "hum": (70, 95), "elev": (0,   800)},
        "cassava":   {"ph": (5.0, 7.0), "rain": (800,  2000), "temp": (20, 35), "hum": (60, 90), "elev": (0,   1200)},
        "cocoyam":   {"ph": (5.5, 7.0), "rain": (1500, 3000), "temp": (20, 28), "hum": (70, 95), "elev": (0,   1500)},
        "plantain":  {"ph": (5.5, 7.0), "rain": (1500, 3500), "temp": (22, 32), "hum": (75, 95), "elev": (0,   1200)},
        "cocoa":     {"ph": (5.5, 7.5), "rain": (1500, 3000), "temp": (18, 28), "hum": (70, 90), "elev": (0,   800)},
        "coffee":    {"ph": (5.5, 7.0), "rain": (1200, 2500), "temp": (15, 24), "hum": (65, 85), "elev": (500, 2000)},
        "groundnut": {"ph": (5.5, 7.0), "rain": (500,  1200), "temp": (25, 35), "hum": (40, 75), "elev": (0,   1000)},
        "beans":     {"ph": (5.5, 7.5), "rain": (800,  1500), "temp": (16, 28), "hum": (55, 80), "elev": (500, 2500)},
        "tomato":    {"ph": (5.5, 7.5), "rain": (600,  1500), "temp": (18, 30), "hum": (50, 80), "elev": (0,   1500)},
        "onion":     {"ph": (5.5, 7.0), "rain": (400,  800),  "temp": (22, 38), "hum": (30, 65), "elev": (0,   1000)},
        "potato":    {"ph": (5.0, 7.0), "rain": (1000, 2000), "temp": (12, 22), "hum": (65, 85), "elev": (800, 3000)},
        "palm_oil":  {"ph": (4.5, 6.5), "rain": (1500, 3500), "temp": (24, 32), "hum": (75, 95), "elev": (0,   500)},
        "sorghum":   {"ph": (5.0, 7.5), "rain": (300,  800),  "temp": (26, 38), "hum": (25, 60), "elev": (0,   1500)},
    }

    for crop, p in crop_profiles.items():
        for _ in range(200):
            records.append({
                "soil_ph":       round(np.random.uniform(*p["ph"]), 1),
                "rainfall_mm":   int(np.random.randint(*p["rain"])),
                "temperature_c": round(np.random.uniform(*p["temp"]), 1),
                "humidity_pct":  round(np.random.uniform(*p["hum"]), 1),
                "elevation_m":   int(np.random.randint(*p["elev"])),
                "crop":          crop,
            })

    df = pd.DataFrame(records)
    X  = df[FEATURES].values
    le = LabelEncoder()
    y  = le.fit_transform(df["crop"].values)
    return X, y, le


# ─── Model training & persistence ─────────────────────────────────────────────

def _train_model(tier: str) -> Tuple[Pipeline, LabelEncoder, Dict]:
    """Train a new suitability model pipeline for the given subscription tier."""
    X, y, le = _get_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if tier == "free":
        clf = DecisionTreeClassifier(max_depth=8, random_state=42)
    elif tier == "medium":
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    else:  # premium
        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

    pipeline = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", clf),
    ])
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "recall":   round(recall_score(y_test, y_pred, average="weighted"), 4),
    }
    return pipeline, le, metrics


def _load_or_train_model(tier: str) -> Tuple[Pipeline, LabelEncoder, Dict]:
    """Load a cached model from disk, or train and cache a fresh one."""
    model_path   = os.path.join(MODEL_DIR, f"suitability_{tier}.pkl")
    le_path      = os.path.join(MODEL_DIR, f"suitability_{tier}_le.pkl")
    metrics_path = os.path.join(MODEL_DIR, f"suitability_{tier}_metrics.pkl")

    if os.path.exists(model_path) and os.path.exists(le_path):
        ml_logger.info("Loading cached suitability model", tier=tier)
        pipeline = joblib.load(model_path)
        le       = joblib.load(le_path)
        metrics  = joblib.load(metrics_path) if os.path.exists(metrics_path) else {}
    else:
        ml_logger.info("Training new suitability model", tier=tier)
        pipeline, le, metrics = _train_model(tier)
        joblib.dump(pipeline, model_path)
        joblib.dump(le,       le_path)
        joblib.dump(metrics,  metrics_path)

    return pipeline, le, metrics


# ─── Live feature fetching ────────────────────────────────────────────────────
async def _fetch_live_features(
    lat: float,
    lon: float,
    region: str,
) -> Tuple[Dict, Dict]:
    """
    Fetch all environmental features from 3rd-party APIs concurrently.

    Sources:
        - Open-Elevation  → elevation_m
        - SoilGrids v2    → soil_ph (+ full texture profile)
        - Open-Meteo      → temperature_c, humidity_pct, rainfall_mm

    Any API failure silently falls back to REGION_ENV_DEFAULTS[region],
    so the model always receives a complete, region-appropriate feature vector.

    Returns:
        env_features : Dict  — the 5 keys consumed by the ML model
        soil_data    : Dict  — full SoilGrids breakdown for the response payload
    """
    region_defaults = REGION_ENV_DEFAULTS.get(region, REGION_ENV_DEFAULTS["Centre"])

    # Initialise with region-aware defaults; overwritten by live data below
    env: Dict = {
        "soil_ph": 6.0,
        "rainfall_mm": float(region_defaults["rainfall_mm"]),
        "temperature_c": float(region_defaults["temperature_c"]),
        "humidity_pct": float(region_defaults["humidity_pct"]),
        "elevation_m": float(region_defaults["elevation_m"]),
    }
    
    # Placeholder for the full SoilGrids breakdown payload mentioned in docstring
    soil_data: Dict = {}

    end_date = (date.today() - timedelta(days=1)).isoformat()
    start_date = (date.today() - timedelta(days=730)).isoformat()
    timeout = httpx.Timeout(15.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        
        # Define isolated, exception-safe fetching routines for asyncio tasks
        async def fetch_elevation():
            try:
                r = await client.get(
                    "https://api.open-elevation.com/api/v1/lookup",
                    params={"locations": f"{lat},{lon}"},
                )
                if r.status_code == 200:
                    env["elevation_m"] = float(r.json()["results"][0]["elevation"])
                else:
                    print(f"Elevation HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                print(f"Elevation Exception: {e}")

        async def fetch_soil():
            try:
                r = await client.get(
                    "https://rest.isric.org/soilgrids/v2.0/properties/query",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "property": ["phh2o"],
                        "depth": ["0-5cm"],
                        "value": ["mean"],
                    },
                )
                if r.status_code == 200:
                    data = r.json()
                    # Store the full raw payload for the response breakdown
                    nonlocal soil_data
                    soil_data = data 
                    
                    layers = data.get("properties", {}).get("layers", [])
                    if layers and "depths" in layers[0]:
                        mean = layers[0]["depths"][0]["values"]["mean"]
                        env["soil_ph"] = round(mean / 10, 2)
                else:
                    print(f"Soil HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                print(f"Soil Exception: {e}")

        async def fetch_weather():
            try:
                r = await client.get(
                    "https://archive-api.open-meteo.com/v1/archive",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start_date,
                        "end_date": end_date,
                        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
                        "timezone": "auto",
                    },
                )
                if r.status_code == 200:
                    daily = r.json().get("daily", {})
                    temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
                    hums = [h for h in daily.get("relative_humidity_2m_mean", []) if h is not None]
                    rains = [p for p in daily.get("precipitation_sum", []) if p is not None]
                    
                    if temps:
                        env["temperature_c"] = round(sum(temps) / len(temps), 2)
                    if hums:
                        env["humidity_pct"] = round(sum(hums) / len(hums), 2)
                    if rains:
                        # Average daily rainfall scaled to an annual total
                        env["rainfall_mm"] = round((sum(rains) / len(rains)) * 365, 2)
                else:
                    print(f"Weather HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                print(f"Weather Exception: {e}")

        # Run all three network API requests concurrently
        await asyncio.gather(
            fetch_elevation(),
            fetch_soil(),
            fetch_weather()
        )
    return env
# ─── Main entry point ─────────────────────────────────────────────────────────

async def run_crop_suitability(
    latitude: float,
    longitude: float,
    subscription_tier: str = "free",
) -> Dict:
    """
    Main entry point for crop suitability analysis.

    Args:
        latitude:          GPS latitude
        longitude:         GPS longitude
        subscription_tier: "free" | "medium" | "premium"

    Returns:
        Complete suitability result dict including crop rankings,
        soil data, environmental readings, and model metadata.
    """
    ml_logger.info(
        "Running crop suitability analysis",
        lat=latitude, lon=longitude, tier=subscription_tier,
    )

    # Resolve region first — needed to pick correct env defaults before API calls
    region = _coords_to_region(latitude, longitude)

    # Fetch live environmental data (with per-region fallbacks)
    env_features = await _fetch_live_features(latitude, longitude, region)

    def _run() -> Dict:
        pipeline, le, metrics = _load_or_train_model(subscription_tier)

        features_array = np.array([[
            env_features["soil_ph"],
            env_features["rainfall_mm"],
            env_features["temperature_c"],
            env_features["humidity_pct"],
            env_features["elevation_m"],
        ]])

        proba        = pipeline.predict_proba(features_array)[0]
        classes      = le.inverse_transform(np.arange(len(proba)))
        scores       = {cls: float(p) for cls, p in zip(classes, proba)}
        sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        n_crops   = {"free": 3, "medium": 7, "premium": len(sorted_crops)}[subscription_tier]
        top_crops = sorted_crops[:n_crops]

        model_name = {
            "free":    "Decision Tree",
            "medium":  "Random Forest",
            "premium": "Gradient Boosting Ensemble",
        }[subscription_tier]

        return {
            "recommended_crop":   top_crops[0][0] if top_crops else "maize",
            "best_crops":         [c for c, _ in top_crops],
            "suitability_scores": dict(top_crops),
            "region_estimate":    region,
            "model_used":         model_name,
            "metrics":            metrics,
            "subscription_tier":  subscription_tier,
            "environmental":      env_features,   # 5 model input features
            "cached":             False,
            "generated_at":       datetime.utcnow().isoformat(),
        }

    # get_running_loop() is correct and deprecation-safe inside async (Python 3.10+)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run)
