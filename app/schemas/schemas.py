"""
Pydantic Schemas — Request/Response Validation
================================================
All API input validation and response serialization schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, validator, field_validator


# ─── Auth Schemas ─────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=200, example="Marie Nguema")
    email: EmailStr = Field(..., example="marie@example.cm")
    password: str = Field(..., min_length=8, max_length=128, example="SecurePass123!")

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: UUID
    full_name: str
    email: str
    subscription_type: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Price Forecast Schemas ───────────────────────────────────────────────────

class PriceForecastRequest(BaseModel):
    duration: int = Field(..., ge=1, le=365, example=2)
    duration_type: str = Field(..., pattern="^(days|weeks|months)$", example="months")

    class Config:
        json_schema_extra = {
            "example": {"duration": 2, "duration_type": "months"}
        }


class PricePredictionPoint(BaseModel):
    date: str
    price_xaf: float = Field(..., description="Price in XAF (CFA Franc)")
    lower_ci: float = Field(..., description="Lower 80% confidence interval")
    upper_ci: float = Field(..., description="Upper 80% confidence interval")


class PriceForecastResponse(BaseModel):
    crop: str
    forecast_period: str
    currency: str = "XAF (CFA Franc)"
    predictions: List[PricePredictionPoint]
    model_used: str
    accuracy: Dict[str, float]
    subscription_tier: str
    cached: bool = False
    generated_at: datetime


# ─── Crop Suitability Schemas ─────────────────────────────────────────────────

class CropSuitabilityRequest(BaseModel):
    latitude: float = Field(..., ge=1.5, le=13.5, example=5.9631)
    longitude: float = Field(..., ge=8.0, le=16.2, example=10.1591)

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 5.9631, "longitude": 10.1591,
            }
        }


class CropSuitabilityResponse(BaseModel):
    best_crops: List[str]
    suitability_scores: Dict[str, float]
    recommended_crop: str
    soil_ph: float
    rainfall: float 
    humidity: float
    temperature: float
    elevation: float
    region_estimate: str
    model_used: str
    metrics: Dict[str, float]
    subscription_tier: str
    cached: bool = False
    generated_at: datetime


# ─── Yield Prediction Schemas ─────────────────────────────────────────────────

class YieldPredictionRequest(BaseModel):
    land_size: float = Field(..., ge=0.1, le=10000, description="Land area in hectares", example=2.5)
    latitude: float = Field(..., ge=1.5, le=13.5, example=5.9631)
    longitude: float = Field(..., ge=8.0, le=16.2, example=10.1591)
    elevation: float = Field(..., ge=0, le=4200, description="Elevation in meters", example=600)
    irrigation: bool = Field(default=False, description="Whether irrigation is used")
    fertilizer_type: Optional[str] = Field(
        default="none",
        pattern="^(none|organic|inorganic|mixed)$",
        example="organic"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "land_size": 2.5, "latitude": 5.9631, "longitude": 10.1591,
                "elevation": 600, "irrigation": False, "fertilizer_type": "organic"
            }
        }


class YieldPredictionResponse(BaseModel):
    crop: str
    predicted_yield: str                    # e.g. "4.3 tons/hectare"
    total_yield_tons: float
    yield_per_hectare: float
    confidence: float
    model_used: str
    metrics: Dict[str, float]
    subscription_tier: str
    cached: bool = False
    generated_at: datetime


# ─── Risk Score Schemas ───────────────────────────────────────────────────────

class RiskScoreRequest(BaseModel):
    crop: str = Field(..., example="maize")
    region: str = Field(..., example="North West")
    land_size: float = Field(..., ge=0.1, le=10000, example=2.5)
    soil_ph: float = Field(..., ge=3.0, le=9.0, example=6.2)
    rainfall: float = Field(..., ge=0, le=5000, example=1800)
    temperature: float = Field(..., ge=10, le=45, example=25)
    market_access: str = Field(
        ...,
        pattern="^(poor|moderate|good|excellent)$",
        description="Market access quality",
        example="moderate"
    )

    @field_validator("crop")
    @classmethod
    def validate_crop(cls, v: str) -> str:
        from app.core.config import settings
        if v.lower() not in settings.SUPPORTED_CROPS:
            raise ValueError(f"Unsupported crop '{v}'. Supported: {settings.SUPPORTED_CROPS}")
        return v.lower()

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        from app.core.config import settings
        valid = [r.lower() for r in settings.CAMEROON_REGIONS]
        if v.lower() not in valid:
            raise ValueError(f"Invalid region '{v}'. Valid regions: {settings.CAMEROON_REGIONS}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "crop": "maize", "region": "North West", "land_size": 2.5,
                "soil_ph": 6.2, "rainfall": 1800, "temperature": 25,
                "market_access": "moderate"
            }
        }


class RiskScoreResponse(BaseModel):
    overall_risk_score: float
    risk_level: str
    financial_risk: float
    climate_risk: float
    agronomic_risk: float
    risk_factors: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]
    subscription_tier: str
    cached: bool = False
    generated_at: datetime


# ─── Report Schemas ───────────────────────────────────────────────────────────

class ReportGenerationRequest(BaseModel):
    analysis_type: str = Field(
        ...,
        pattern="^(price_forecast|crop_suitability|yield_prediction|risk_score)$"
    )
    result_id: UUID
    format: str = Field(default="json", pattern="^(json|pdf)$")


class ReportStatusResponse(BaseModel):
    report_id: UUID
    status: str
    format: str
    is_ready: bool
    download_url: Optional[str] = None
    created_at: datetime


# ─── General Schemas ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    services: Dict[str, str]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    per_page: int
    pages: int
