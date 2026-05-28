"""
Dataset Generation Script
==========================
Generates realistic synthetic crop price datasets for Cameroon.
Based on real market price ranges, seasonality patterns, and
economic factors specific to Cameroon agriculture.

All prices in XAF (CFA Franc) per kg.
Run: python scripts/generate_datasets.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ─── Seed for reproducibility ────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ─── Output directories ───────────────────────────────────────────────────────
PRICES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "prices")
YIELDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "yields")
SOIL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "soil")
CLIMATE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "climate")

for d in [PRICES_DIR, YIELDS_DIR, SOIL_DIR, CLIMATE_DIR]:
    os.makedirs(d, exist_ok=True)


# ─── Crop Price Configuration ─────────────────────────────────────────────────
# Prices in XAF/kg — sourced from MINADER, FAO Cameroon reports
CROP_PRICE_CONFIG = {
    "maize": {
        "base_price": 250,        # XAF/kg
        "seasonality_amp": 0.35,  # 35% seasonal swing
        "trend_rate": 0.02,       # 2% annual growth
        "volatility": 0.08,
        "peak_month": 8,          # Lean season peak
        "trough_month": 11,       # Harvest trough
    },
    "rice": {
        "base_price": 450,
        "seasonality_amp": 0.20,
        "trend_rate": 0.03,
        "volatility": 0.10,
        "peak_month": 6,
        "trough_month": 12,
    },
    "cassava": {
        "base_price": 120,
        "seasonality_amp": 0.25,
        "trend_rate": 0.015,
        "volatility": 0.07,
        "peak_month": 4,
        "trough_month": 9,
    },
    "cocoyam": {
        "base_price": 200,
        "seasonality_amp": 0.30,
        "trend_rate": 0.02,
        "volatility": 0.09,
        "peak_month": 5,
        "trough_month": 10,
    },
    "plantain": {
        "base_price": 150,
        "seasonality_amp": 0.45,
        "trend_rate": 0.025,
        "volatility": 0.12,
        "peak_month": 7,
        "trough_month": 11,
    },
    "cocoa": {
        "base_price": 1200,
        "seasonality_amp": 0.15,
        "trend_rate": 0.04,
        "volatility": 0.18,
        "peak_month": 10,
        "trough_month": 3,
    },
    "coffee": {
        "base_price": 900,
        "seasonality_amp": 0.20,
        "trend_rate": 0.03,
        "volatility": 0.15,
        "peak_month": 11,
        "trough_month": 5,
    },
    "groundnut": {
        "base_price": 600,
        "seasonality_amp": 0.28,
        "trend_rate": 0.02,
        "volatility": 0.10,
        "peak_month": 4,
        "trough_month": 10,
    },
    "beans": {
        "base_price": 700,
        "seasonality_amp": 0.35,
        "trend_rate": 0.02,
        "volatility": 0.11,
        "peak_month": 3,
        "trough_month": 9,
    },
    "tomato": {
        "base_price": 300,
        "seasonality_amp": 0.60,
        "trend_rate": 0.03,
        "volatility": 0.20,
        "peak_month": 6,
        "trough_month": 12,
    },
    "onion": {
        "base_price": 400,
        "seasonality_amp": 0.50,
        "trend_rate": 0.025,
        "volatility": 0.18,
        "peak_month": 5,
        "trough_month": 11,
    },
    "potato": {
        "base_price": 280,
        "seasonality_amp": 0.35,
        "trend_rate": 0.02,
        "volatility": 0.12,
        "peak_month": 5,
        "trough_month": 10,
    },
    "palm_oil": {
        "base_price": 1000,
        "seasonality_amp": 0.12,
        "trend_rate": 0.035,
        "volatility": 0.08,
        "peak_month": 3,
        "trough_month": 8,
    },
    "sorghum": {
        "base_price": 220,
        "seasonality_amp": 0.30,
        "trend_rate": 0.015,
        "volatility": 0.09,
        "peak_month": 7,
        "trough_month": 11,
    },
}


def generate_price_series(crop: str, config: dict, years: int = 8) -> pd.DataFrame:
    """
    Generate a realistic weekly price time series for a crop.

    Models:
    - Long-term trend (economic growth)
    - Annual seasonality (harvest cycles)
    - Random shocks (weather events, policy changes)
    - ARMA-like autocorrelation
    """
    start_date = datetime(2016, 1, 1)
    periods = years * 52  # weekly data

    dates = [start_date + timedelta(weeks=i) for i in range(periods)]
    prices = []
    regions = ["North West", "South West", "West", "Adamawa", "Far North",
               "Centre", "East", "North", "South", "Littoral"]

    price = config["base_price"]

    for i, date in enumerate(dates):
        # Trend component
        years_elapsed = i / 52
        trend = config["base_price"] * (1 + config["trend_rate"]) ** years_elapsed

        # Seasonality component (sinusoidal harvest cycle)
        month = date.month
        seasonal_phase = 2 * np.pi * (month - config["trough_month"]) / 12
        seasonal = 1 + config["seasonality_amp"] * np.sin(seasonal_phase)

        # Random shock (occasional drought, flood, policy)
        shock = 1.0
        if random.random() < 0.02:  # 2% chance of shock event
            shock = 1 + random.uniform(-0.25, 0.35)

        # White noise
        noise = np.random.normal(1.0, config["volatility"])

        # AR(1) component — price persistence
        price = 0.7 * price + 0.3 * (trend * seasonal * shock * noise)
        price = max(price, config["base_price"] * 0.3)  # Floor price

        for region in regions:
            # Regional price variation (transport costs, local supply)
            regional_multiplier = np.random.uniform(0.88, 1.15)
            regional_price = price * regional_multiplier

            prices.append({
                "date": date.strftime("%Y-%m-%d"),
                "price_xaf_per_kg": round(regional_price, 0),
                "region": region,
                "crop": crop,
                "year": date.year,
                "month": date.month,
                "week": date.isocalendar()[1],
            })

    return pd.DataFrame(prices)


def generate_yield_dataset() -> pd.DataFrame:
    """
    Generate a crop yield dataset for Cameroon regions.
    Yields in tons/hectare — based on FAOSTAT Cameroon data.
    """
    crops_yields = {
        "maize":     {"base": 1.8,  "std": 0.4,  "with_irrigation": 2.8,  "with_fertilizer": 2.5},
        "rice":      {"base": 2.2,  "std": 0.5,  "with_irrigation": 4.0,  "with_fertilizer": 3.2},
        "cassava":   {"base": 10.0, "std": 2.0,  "with_irrigation": 14.0, "with_fertilizer": 13.0},
        "cocoyam":   {"base": 5.0,  "std": 1.2,  "with_irrigation": 7.0,  "with_fertilizer": 6.5},
        "plantain":  {"base": 8.0,  "std": 1.5,  "with_irrigation": 11.0, "with_fertilizer": 10.0},
        "cocoa":     {"base": 0.45, "std": 0.1,  "with_irrigation": 0.60, "with_fertilizer": 0.55},
        "coffee":    {"base": 0.35, "std": 0.08, "with_irrigation": 0.50, "with_fertilizer": 0.45},
        "groundnut": {"base": 1.2,  "std": 0.3,  "with_irrigation": 1.8,  "with_fertilizer": 1.6},
        "beans":     {"base": 0.8,  "std": 0.2,  "with_irrigation": 1.3,  "with_fertilizer": 1.1},
        "tomato":    {"base": 15.0, "std": 3.0,  "with_irrigation": 30.0, "with_fertilizer": 22.0},
        "onion":     {"base": 12.0, "std": 2.5,  "with_irrigation": 20.0, "with_fertilizer": 17.0},
        "potato":    {"base": 9.0,  "std": 2.0,  "with_irrigation": 15.0, "with_fertilizer": 13.0},
        "palm_oil":  {"base": 3.5,  "std": 0.7,  "with_irrigation": 5.0,  "with_fertilizer": 4.5},
        "sorghum":   {"base": 1.0,  "std": 0.25, "with_irrigation": 1.6,  "with_fertilizer": 1.4},
    }

    regions_climate = {
        "North West":  {"rainfall": 1800, "temp": 19, "humidity": 78, "elevation": 1500},
        "South West":  {"rainfall": 3500, "temp": 25, "humidity": 85, "elevation": 400},
        "West":        {"rainfall": 1600, "temp": 20, "humidity": 75, "elevation": 1200},
        "Littoral":    {"rainfall": 2800, "temp": 27, "humidity": 82, "elevation": 50},
        "Adamawa":     {"rainfall": 1200, "temp": 22, "humidity": 60, "elevation": 900},
        "Far North":   {"rainfall": 500,  "temp": 32, "humidity": 35, "elevation": 300},
        "Centre":      {"rainfall": 1600, "temp": 25, "humidity": 78, "elevation": 700},
        "East":        {"rainfall": 1700, "temp": 24, "humidity": 80, "elevation": 600},
        "North":       {"rainfall": 800,  "temp": 30, "humidity": 45, "elevation": 400},
        "South":       {"rainfall": 1800, "temp": 26, "humidity": 83, "elevation": 500},
    }

    records = []
    for i in range(3000):
        crop = random.choice(list(crops_yields.keys()))
        region = random.choice(list(regions_climate.keys()))
        climate = regions_climate[region]
        yield_config = crops_yields[crop]

        land_size = round(np.random.uniform(0.5, 50), 2)
        soil_ph = round(np.random.uniform(4.5, 7.5), 1)
        irrigation = random.choice([True, False])
        fertilizer = random.choice(["none", "organic", "inorganic", "mixed"])

        # Calculate expected yield
        base_yield = yield_config["base"]
        if irrigation:
            base_yield = yield_config["with_irrigation"]
        if fertilizer != "none":
            base_yield = max(base_yield, yield_config["with_fertilizer"])

        # Climate adjustments
        rainfall_factor = min(climate["rainfall"] / 1500, 1.3)
        temp_factor = 1.0 - abs(climate["temp"] - 25) * 0.01
        ph_factor = 1.0 - abs(soil_ph - 6.2) * 0.05

        actual_yield = base_yield * rainfall_factor * temp_factor * ph_factor
        actual_yield += np.random.normal(0, yield_config["std"] * 0.3)
        actual_yield = max(actual_yield, 0.1)

        records.append({
            "crop": crop,
            "region": region,
            "land_size_ha": land_size,
            "soil_ph": soil_ph,
            "rainfall_mm": climate["rainfall"] + np.random.randint(-200, 200),
            "temperature_c": climate["temp"] + np.random.uniform(-2, 2),
            "humidity_pct": climate["humidity"] + np.random.uniform(-10, 10),
            "elevation_m": climate["elevation"] + np.random.randint(-100, 100),
            "irrigation": int(irrigation),
            "fertilizer_type": fertilizer,
            "yield_tons_per_ha": round(actual_yield, 3),
            "total_yield_tons": round(actual_yield * land_size, 3),
            "year": random.randint(2016, 2024),
        })

    return pd.DataFrame(records)


def generate_soil_dataset() -> pd.DataFrame:
    """Soil characteristics for Cameroon agricultural zones."""
    soil_types = ["Clay", "Sandy loam", "Loam", "Sandy clay", "Silty clay", "Laterite"]

    records = []
    for _ in range(1000):
        records.append({
            "latitude": round(np.random.uniform(2.0, 13.0), 4),
            "longitude": round(np.random.uniform(8.5, 16.0), 4),
            "soil_type": random.choice(soil_types),
            "soil_ph": round(np.random.uniform(4.0, 8.0), 1),
            "organic_matter_pct": round(np.random.uniform(0.5, 8.0), 2),
            "nitrogen_pct": round(np.random.uniform(0.02, 0.5), 3),
            "phosphorus_ppm": round(np.random.uniform(5, 80), 1),
            "potassium_meq": round(np.random.uniform(0.1, 2.0), 2),
            "cec": round(np.random.uniform(5, 40), 1),
            "drainage": random.choice(["poor", "moderate", "good", "excessive"]),
            "depth_cm": random.randint(30, 200),
        })

    return pd.DataFrame(records)


def main():
    print("🌱 Generating Cameroon Agricultural Datasets...\n")

    # ─── Price Datasets ───────────────────────────────────────────────────────
    print("📊 Generating crop price time series...")
    for crop, config in CROP_PRICE_CONFIG.items():
        df = generate_price_series(crop, config)
        filepath = os.path.join(PRICES_DIR, f"{crop}_price.csv")
        df.to_csv(filepath, index=False)
        print(f"  ✅ {crop}_price.csv — {len(df)} rows")

    # ─── Yield Dataset ────────────────────────────────────────────────────────
    print("\n🌾 Generating yield dataset...")
    yield_df = generate_yield_dataset()
    yield_df.to_csv(os.path.join(YIELDS_DIR, "cameroon_yields.csv"), index=False)
    print(f"  ✅ cameroon_yields.csv — {len(yield_df)} rows")

    # ─── Soil Dataset ─────────────────────────────────────────────────────────
    print("\n🪨 Generating soil dataset...")
    soil_df = generate_soil_dataset()
    soil_df.to_csv(os.path.join(SOIL_DIR, "cameroon_soil.csv"), index=False)
    print(f"  ✅ cameroon_soil.csv — {len(soil_df)} rows")

    print("\n🎉 All datasets generated successfully!")
    print(f"   Prices:  {PRICES_DIR}")
    print(f"   Yields:  {YIELDS_DIR}")
    print(f"   Soil:    {SOIL_DIR}")


if __name__ == "__main__":
    main()
