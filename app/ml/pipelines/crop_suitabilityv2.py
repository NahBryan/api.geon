
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
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import httpx  
import asyncio
import geopandas as gpd  # Added for Shapefile manipulation
from shapely.geometry import Point  # Added for Point-in-Polygon intersections
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

from app.core.config import settings
from app.core.logging import ml_logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml_models", "suitability")
YIELDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "yields")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "gadm41_CMR_shp")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Features used for suitability prediction ────────────────────────────────
FEATURES = ["soil_ph", "rainfall_mm", "temperature_c", "humidity_pct", "elevation_m"]
TARGET = "crop"
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ─── Load GADM Shapefile for Cameroon (Level 1 - Regions) ────────────────────
SHAPEFILE_PATH = os.path.join(DATA_DIR, "gadm41_CMR_1.shp")

try:
    if os.path.exists(SHAPEFILE_PATH):
        # Read shapefile and ensure it works with standard GPS coordinates (WGS84)
        CMR_REGIONS_GDF = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        ml_logger.info("Successfully loaded GADM Cameroon shapefile boundaries.")
    else:
        CMR_REGIONS_GDF = None
        ml_logger.warning(f"GADM Shapefile not found at {SHAPEFILE_PATH}. Falling back to default region values.")
except Exception as e:
    CMR_REGIONS_GDF = None
    ml_logger.error(f"Failed to load GADM Shapefile: {e}")


def _coords_to_region(lat: float, lon: float) -> str:
    """
    Infer official Cameroon region from GPS coordinates using GADM polygons.
    """
    if CMR_REGIONS_GDF is None:
        return "Centre"  # Safe programmatic fallback if file isn't present yet

    try:
        # Create a geometric point from input coordinates (lon, lat order for shapely)
        point = Point(lon, lat)
        
        # Check which polygon contains the point
        match = CMR_REGIONS_GDF[CMR_REGIONS_GDF.geometry.contains(point)]
        
        if not match.empty:
            # 'NAME_1' is the standard GADM attribute column for Level-1 administrative regions
            return match.iloc[0]['NAME_1']
    except Exception as e:
        ml_logger.error(f"Error resolving coordinate to GADM region: {e}")
        
    return "Centre"  # Default fallback if coordinates fall slightly outside borders


def _get_training_data() -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load yield dataset and prepare suitability training data."""
    filepath = os.path.join(YIELDS_DIR, "cameroon_yields.csv")
    if not os.path.exists(filepath):
        return _generate_synthetic_training_data()

    df = pd.read_csv(filepath)
    X = df[FEATURES].values
    y = df[TARGET].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le


def _generate_synthetic_training_data() -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Generate synthetic training data based on known Cameroon crop requirements."""
    np.random.seed(42)
    records = []

    crop_profiles = {
        "maize":     {"ph": (5.5, 7.5), "rain": (600, 1200),  "temp": (18, 32), "hum": (50, 85), "elev": (0, 2000)},
        "rice":      {"ph": (5.0, 7.0), "rain": (1200, 2500), "temp": (22, 35), "hum": (70, 95), "elev": (0, 800)},
        "cassava":   {"ph": (5.0, 7.0), "rain": (800, 2000),  "temp": (20, 35), "hum": (60, 90), "elev": (0, 1200)},
        "cocoyam":   {"ph": (5.5, 7.0), "rain": (1500, 3000), "temp": (20, 28), "hum": (70, 95), "elev": (0, 1500)},
        "plantain":  {"ph": (5.5, 7.0), "rain": (1500, 3500), "temp": (22, 32), "hum": (75, 95), "elev": (0, 1200)},
        "cocoa":     {"ph": (5.5, 7.5), "rain": (1500, 3000), "temp": (18, 28), "hum": (70, 90), "elev": (0, 800)},
        "coffee":    {"ph": (5.5, 7.0), "rain": (1200, 2500), "temp": (15, 24), "hum": (65, 85), "elev": (500, 2000)},
        "groundnut": {"ph": (5.5, 7.0), "rain": (500, 1200),  "temp": (25, 35), "hum": (40, 75), "elev": (0, 1000)},
        "beans":     {"ph": (5.5, 7.5), "rain": (800, 1500),  "temp": (16, 28), "hum": (55, 80), "elev": (500, 2500)},
        "tomato":    {"ph": (5.5, 7.5), "rain": (600, 1500),  "temp": (18, 30), "hum": (50, 80), "elev": (0, 1500)},
        "onion":     {"ph": (5.5, 7.0), "rain": (400, 800),   "temp": (22, 38), "hum": (30, 65), "elev": (0, 1000)},
        "potato":    {"ph": (5.0, 7.0), "rain": (1000, 2000), "temp": (12, 22), "hum": (65, 85), "elev": (800, 3000)},
        "palm_oil":  {"ph": (4.5, 6.5), "rain": (1500, 3500), "temp": (24, 32), "hum": (75, 95), "elev": (0, 500)},
        "sorghum":   {"ph": (5.0, 7.5), "rain": (300, 800),   "temp": (26, 38), "hum": (25, 60), "elev": (0, 1500)},
    }

    for crop, profile in crop_profiles.items():
        for _ in range(200):
            records.append({
                "soil_ph": round(np.random.uniform(*profile["ph"]), 1),
                "rainfall_mm": np.random.randint(*profile["rain"]),
                "temperature_c": round(np.random.uniform(*profile["temp"]), 1),
                "humidity_pct": round(np.random.uniform(*profile["hum"]), 1),
                "elevation_m": np.random.randint(*profile["elev"]),
                "crop": crop,
            })

    df = pd.DataFrame(records)
    X = df[FEATURES].values
    le = LabelEncoder()
    y = le.fit_transform(df["crop"].values)
    return X, y, le


def _train_model(tier: str) -> Tuple[Pipeline, LabelEncoder, Dict[str, float]]:
    """Train suitability model for a given tier."""
    X, y, le = _get_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if tier == "free":
        clf = DecisionTreeClassifier(max_depth=8, random_state=42)
    elif tier == "medium":
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    else:
        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
    }

    return pipeline, le, metrics


def _load_or_train_model(tier: str):
    """Load cached model or train fresh if not available."""
    model_path = os.path.join(MODEL_DIR, f"suitability_{tier}.pkl")
    le_path = os.path.join(MODEL_DIR, f"suitability_{tier}_le.pkl")
    metrics_path = os.path.join(MODEL_DIR, f"suitability_{tier}_metrics.pkl")

    if os.path.exists(model_path) and os.path.exists(le_path):
        ml_logger.info("Loading cached suitability model", tier=tier)
        pipeline = joblib.load(model_path)
        le = joblib.load(le_path)
        metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else {}
    else:
        ml_logger.info("Training new suitability model", tier=tier)
        pipeline, le, metrics = _train_model(tier)
        joblib.dump(pipeline, model_path)
        joblib.dump(le, le_path)
        joblib.dump(metrics, metrics_path)

    return pipeline, le, metrics


async def _fetch_live_features(lat: float, lon: float) -> Dict[str, float]:
    """Fetch environmental attributes asynchronously via 3rd-party APIs."""
    features = {"soil_ph": 6.0, "rainfall": 1200.0, "temperature": 24.0, "humidity": 70.0, "elevation_m": 500.0}
    timeout = httpx.Timeout(10.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            client.get(f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"),
            client.get(f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&depth=0-5cm&value=mean"),
            client.get(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2023-01-01&end_date=2025-12-31&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean&timezone=auto")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        if not isinstance(results[0], Exception) and results[0].status_code == 200:
            try:
                features["elevation"] = float(results[0].json()['results'][0]['elevation'])
            except (KeyError, IndexError, ValueError):
                pass
                
        if not isinstance(results[1], Exception) and results[1].status_code == 200:
            try:
                raw_ph = results[1].json()['properties']['layers'][0]['depths'][0]['values']['mean']
                features["soil_ph"] = round(float(raw_ph) / 10.0, 2)
            except (KeyError, IndexError, ValueError):
                pass

        if not isinstance(results[2], Exception) and results[2].status_code == 200:
            try:
                daily = results[2].json()['daily']
                temps = [t for t in daily['temperature_2m_mean'] if t is not None]
                if temps:
                    features["temperature"] = round(sum(temps) / len(temps), 2)
                
                hums = [h for h in daily['relative_humidity_2m_mean'] if h is not None]
                if hums:
                    features["humidity"] = round(sum(hums) / len(hums), 2)
                
                rains = [r for r in daily['precipitation_sum'] if r is not None]
                if rains:
                    features["rainfall"] = round((sum(rains) / len(rains)) * 365, 2)
            except (KeyError, ValueError):
                pass

    return features


async def run_crop_suitability(
    latitude: float,
    longitude: float,
    subscription_tier: str = "free",
) -> Dict:
    """
    Main entry point for crop suitability analysis.
    """
    ml_logger.info(
        "Running live crop suitability analysis",
        lat=latitude, lon=longitude, tier=subscription_tier
    )

    live_data = await _fetch_live_features(latitude, longitude)

    def _run():
        pipeline, le, metrics = _load_or_train_model(subscription_tier)

        features_array = np.array([[
            live_data["soil_ph"],
            live_data["rainfall"],
            live_data["temperature"],
            live_data["humidity"],
            live_data["elevation"]
        ]])
        
        proba = pipeline.predict_proba(features_array)[0]
        classes = le.inverse_transform(np.arange(len(proba)))

        scores = {cls: round(float(prob), 4) for cls, prob in zip(classes, proba)}
        sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        n_crops = {"free": 3, "medium": 7, "premium": len(sorted_crops)}[subscription_tier]
        top_crops = sorted_crops[:n_crops]

        # Call the updated shapefile coordinate resolver
        region = _coords_to_region(latitude, longitude)

        model_name = {
            "free": "Decision Tree",
            "medium": "Random Forest",
            "premium": "Gradient Boosting Ensemble",
        }[subscription_tier]

        return {
            "best_crops": [c for c, _ in top_crops],
            "suitability_scores": dict(top_crops),
            "recommended_crop": top_crops[0][0] if top_crops else "maize",
            "region_estimate": region,
            "model_used": model_name,
            "metrics": metrics,
            "subscription_tier": subscription_tier,
            "environmental": live_data,
            "cached": False,
            "generated_at": datetime.utcnow().isoformat(),
        }

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)
