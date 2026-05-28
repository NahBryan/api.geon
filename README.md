# *Project Title: AGRO-CLIMATIC INTELLIGENCE SYSTEM (GEON)*
# Author: NAH BRYAN MIKE CHIDOH
# Degree: Bachelor of Technology (B-Tech) Defense Project
# Status: Academic Research (Not for Commercial Sale)

---
## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Subscription Plans](#subscription-plans)
- [Request Deduplication](#request-deduplication)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)

---

# 1. Project Overview
The Agro-Climatic Intelligence System (GEON)  is an R-based analytical backend designed to provide data-driven insights for the Cameroonian agricultural sector. The system utilizes data mining and predictive modeling to assist farmers, researchers, and policy-makers in two critical areas:
Crop Suitability Analysis: Evaluating soil and climatic parameters across Cameroon's five agroecological zones.
Price Forecasting: Utilizing historical trends to predict market fluctuations for staple and cash crops.
The project is structured as a modular API-ready service, ensuring scalability and integration with modern web or mobile frontends.

# 2. Scientific Basis & Calibration
This system is calibrated using verified peer-reviewed research and official agricultural databases specific to the Cameroonian context. Key data sources include:
Soil Nutrient Benchmarks: Nanganoa et al. (2020) - Soil Nutrients in AEZs of Cameroon (IRAD).
Regional Profiles: IRAD Agroecological Zone profiles (2009).
Crop Requirements: FAO Crop Requirements Database.
Economic Trends: Mordor Intelligence - Crops in Cameroon Market 2024.
Soil Mapping: iSDAsoil Africa soil property benchmarks.

# 3. Geographic & Agricultural Scope
# 3.1 Agroecological Zones (AEZ) Coverage
The model accounts for the unique characteristics of Cameroon's five distinct zones:
AEZ 1 (Sudano-Sahelian): Hot/Dry, ~800mm rain, 29°C (Far North/North).
AEZ 2 (High Guinea Savanna): ~1200-1600mm rain, 24°C (Adamawa/North).
AEZ 3 (Western Highlands): ~1800mm rain, 18-22°C, acidic soils (West/NW).
AEZ 4 (Humid Forest Monomodal): ~2500mm+ rain, 24°C (SW/Littoral).
AEZ 5 (Humid Forest Bimodal): ~1600-2000mm rain, 25°C (Centre/South/East).

# 3.2 Supported Crops (20 Species)
Staples: Maize, Cassava, Sorghum, Millet, Rice, Yam, Cocoyam, Sweet Potato, Plantain.
Legumes: Groundnut, Cowpea, Soybean, Beans.
Cash Crops: Cocoa, Coffee, Oil Palm, Cotton, Sugarcane.
Vegetables: Tomato, Cabbage.
 AI-powered agricultural analytics for Cameroon farmers, cooperatives, and agribusinesses.

## Overview

CamAgri provides four core ML-powered capabilities:

| Feature | Description | Models |
|---------|-------------|--------|
| **Price Forecasting** | Predict future crop prices in XAF | ARIMA / Prophet / Ensemble |
| **Crop Suitability** | Best crops for a given location | Decision Tree / Random Forest / GBM |
| **Yield Prediction** | Harvest estimation per hectare | Linear Regression / RF / GBM |
| **Risk Scoring** | Financial + climate + agronomic risk | Rule-based + ML ensemble |

**Supported Crops:** Maize, Rice, Cassava, Cocoyam, Plantain, Cocoa, Coffee, Groundnut, Beans, Tomato, Onion, Potato, Palm Oil, Sorghum

**Supported Regions:** All 10 Cameroon regions (North West, South West, West, Littoral, Adamawa, Far North, Centre, East, North, South)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client (Browser/App)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTPS
┌─────────────────────▼───────────────────────────────────────┐
│              FastAPI Application (Port 8000)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Middleware: CORS → Error Handler → Rate Limiter      │   │
│  │             → Request Logger → JWT Auth              │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌───────────────┐ ┌──────────────┐ ┌──────────────────┐   │
│  │  /auth/*      │ │  /analysis/* │ │  /reports/*      │   │
│  └───────────────┘ └──────┬───────┘ └──────────────────┘   │
│                            │                                  │
│  ┌─────────────────────────▼────────────────────────────┐   │
│  │           Cache-or-Compute Layer                      │   │
│  │  1. Redis cache check (fast, ~1ms)                    │   │
│  │  2. PostgreSQL hash check (persistent)                │   │
│  │  3. ML Pipeline (compute only on cache miss)          │   │
│  └──────┬──────────────────────────────────────────────┘   │
└─────────┼────────────────────────────────────────────────────┘
          │
     ┌────▼──────────────────────────────────────────┐
     │              ML Pipelines                       │
     │  ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
     │  │  ARIMA   │ │ Prophet  │ │    Ensemble    │ │
     │  │          │ │          │ │  (ARIMA+Proph) │ │
     │  └──────────┘ └──────────┘ └────────────────┘ │
     │  ┌──────────────────────────────────────────┐  │
     │  │ Random Forest  │  Gradient Boosting      │  │
     │  └──────────────────────────────────────────┘  │
     └───────────────────────────────────────────────┘
          │                    │
     ┌────▼──────┐      ┌─────▼──────┐
     │ PostgreSQL│      │   Redis    │
     │ (results) │      │ (cache +   │
     │           │      │  rate lim) │
     └───────────┘      └────────────┘
          │
     ┌────▼────────────────────────────┐
     │      Celery Workers (async)      │
     │  - Model retraining (weekly)     │
     │  - Cache warming (daily)         │
     │  - Report generation             │
     └──────────────────────────────────┘
```

---

## Quick Start

### Docker (Recommended)

```bash
# 1. Clone and configure
git clone https://github.com/your-org/camagri-platform.git
cd camagri-platform
cp .env.example .env
# Edit .env — set SECRET_KEY and other values

# 2. Generate datasets
pip install pandas numpy
python scripts/generate_datasets.py #use this only for testing check the dataset folder if datasets exists

# 3. Start all services
docker-compose up -d

# 4. Run database migrations
docker-compose exec api alembic upgrade head

# 5. Open API docs
open http://localhost:8000/docs
```

Services available:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Flower (Celery)**: http://localhost:5555
- **Prometheus**: http://localhost:9090
- **PgAdmin** (dev only): http://localhost:5050

---

## API Reference

### Authentication

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"full_name":"Jean Paul","email":"jp@test.cm","password":"SecurePass123!"}'

# Login → get tokens
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"jp@test.cm","password":"SecurePass123!"}'
```

### Price Forecasting

```bash
curl -X POST http://localhost:8000/api/v1/price-forecasting/maize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"duration": 2, "duration_type": "months"}'
```

**Response:**
```json
{
  "crop": "maize",
  "forecast_period": "2 months",
  "currency": "XAF (CFA Franc)",
  "predictions": [
    {"date": "2024-06-01", "price_xaf": 265, "lower_ci": 240, "upper_ci": 290}
  ],
  "model_used": "Prophet",
  "accuracy": {"rmse": 12.5, "mape": 4.8, "forecast_bias": 0.02},
  "subscription_tier": "medium",
  "cached": false
}
```

### Crop Suitability

```bash
curl -X POST http://localhost:8000/api/v1/crop-suitability \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 5.9631, "longitude": 10.1591,
    "soil_ph": 6.2, "rainfall": 1800,
    "temperature": 25, "humidity": 75, "elevation": 600
  }'
```

### Yield Prediction

```bash
curl -X POST http://localhost:8000/api/v1/yield-prediction/maize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "land_size": 2.5, "soil_ph": 6.2, "rainfall": 1800,
    "temperature": 25, "humidity": 75, "elevation": 600,
    "irrigation": false, "fertilizer_type": "organic"
  }'
```

### Risk Score

```bash
curl -X POST http://localhost:8000/api/v1/risk-score \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "maize", "region": "North West", "land_size": 2.5,
    "soil_ph": 6.2, "rainfall": 1800, "temperature": 25,
    "market_access": "moderate"
  }'
```

---

## Subscription Plans

| | FREE | MEDIUM | PREMIUM |
|---|---|---|---|
| Daily Requests | 10 | 100 | 10,000 |
| Price Forecast Model | ARIMA | Prophet | Ensemble |
| Suitability Model | Decision Tree | Random Forest | Gradient Boosting |
| Yield Model | Linear Regression | Random Forest | GBM |
| Report Formats | JSON | JSON + PDF | JSON + PDF |
| Forecast Granularity | Monthly | Weekly | Daily |

---

## Request Deduplication

**This is a core feature.** Identical requests never recompute — results are returned from cache.

A SHA-256 hash is generated from all request parameters + subscription tier:

```python
hash_price_forecast("rice", 2, "months", "premium")
# → "a3f8c2d1..." (always the same for same inputs)
```

Cache lookup order:
1. **Redis** (in-memory, ~1ms) — TTL varies by tier
2. **PostgreSQL** (persistent) — survives Redis restart
3. **ML Compute** (only if both miss)

---

## Development Setup

```bash
# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start dependencies
docker-compose up postgres redis -d

# Generate datasets
python scripts/generate_datasets.py

# Run migrations
alembic upgrade head

# Start dev server
uvicorn app.main:app --reload --port 8000
```

---

## Running Tests

```bash
pytest tests/ -v --cov=app
```

---

## Project Structure

```
camagri-platform/
├── app/
│   ├── api/v1/endpoints/    # Route handlers
│   │   ├── auth.py          # Register, Login, Refresh, Me
│   │   ├── analysis.py      # Price, Suitability, Yield, Risk
│   │   ├── reports.py       # Report generation & download
│   │   └── health.py        # Health checks
│   ├── core/
│   │   ├── config.py        # Settings (Pydantic)
│   │   ├── security.py      # JWT + bcrypt
│   │   ├── hashing.py       # Request fingerprinting
│   │   └── logging.py       # Structlog setup
│   ├── ml/pipelines/
│   │   ├── price_forecasting.py    # ARIMA + Prophet + Ensemble
│   │   ├── crop_suitability.py     # RF + DT + GBM classifiers
│   │   └── yield_and_risk.py       # Regressors + risk scoring
│   ├── models/models.py     # SQLAlchemy ORM models
│   ├── schemas/schemas.py   # Pydantic request/response schemas
│   ├── services/cache.py    # Redis caching layer
│   ├── middleware/          # CORS, rate limiting, logging, errors
│   ├── workers/tasks.py     # Celery background tasks
│   └── main.py              # FastAPI app factory
├── data/
│   ├── prices/              # {crop}_price.csv (14 files)
│   ├── yields/              # cameroon_yields.csv
│   └── soil/                # cameroon_soil.csv
├── ml_models/               # Persisted .pkl model files
├── scripts/
│   ├── generate_datasets.py # Dataset generation for testing
│   └── init_db.sql          # PostgreSQL initialization
├── tests/test_all.py        # Full test suite
├── monitoring/prometheus.yml
├── alembic/                 # DB migrations
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

# 5. Technical Implementation
Language: R (utilizing Plumber for API routing).
Security: * JWT: Secure authentication for multi-tenant access.
RLS (Row-Level Security): Data isolation at the database layer ensuring user/tenant data privacy.
Rate Limiting: Protects the system from automated scraping.
Analytics: Implementation of FAO-standard crop-matching algorithms and time-series forecasting.

# 6. Academic Disclaimer
This project is submitted in partial fulfillment of the requirements for the Bachelor of Technology (B-Tech) degree. It is intended for academic defense and research purposes only. No part of this software is intended for commercial sale or distribution without the express written consent of the author.

# 7. Setup & Execution
Ensure R and the plumber library are installed.
Configure your environment variables for database connection in R/db.R.
Run the application:
    -Highlight the main.R and hit *ctrl + Enter* or``` source('main.R')```
The API will be available at http://localhost:5000 (default).
