"""
ORM Models — Users & Authentication
=====================================
User accounts, subscriptions, and audit trail.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, ForeignKey,
    Integer, String, Text, Float, JSON, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base


def utcnow():
    return datetime.now(timezone.utc)


# ─── Enums ────────────────────────────────────────────────────────────────────

class SubscriptionType(str, PyEnum):
    free = "free"
    medium = "medium"
    premium = "premium"


class RequestStatus(str, PyEnum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cached = "cached"


class ReportFormat(str, PyEnum):
    JSON = "json"
    PDF = "pdf"


class RiskLevel(str, PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ─── User ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(200), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    subscription_type = Column(Enum(SubscriptionType, name="subscription_type"), default=SubscriptionType.free, nullable=False
    )
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    role = Column(String(50), default="user", nullable=False)  # user | admin
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    requests = relationship("RequestLog", back_populates="user", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")

    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
    )


# ─── Subscription ─────────────────────────────────────────────────────────────

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    plan = Column(Enum(SubscriptionType), nullable=False)
    started_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    payment_reference = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# ─── Request Deduplication Log ────────────────────────────────────────────────

class RequestLog(Base):
    """
    Tracks all API requests with their parameter hash.
    When hash matches, return cached result — no recomputation.
    """
    __tablename__ = "requests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    request_hash = Column(String(64), nullable=False, index=True)
    endpoint = Column(String(100), nullable=False)
    parameters = Column(JSON, nullable=False)
    status = Column(Enum(RequestStatus, name="request_status"), default=RequestStatus.pending, nullable=False)
    result_id = Column(UUID(as_uuid=True), nullable=True)   # FK to result table
    result_table = Column(String(50), nullable=True)         # Table name for result
    processing_time_ms = Column(Integer, nullable=True)
    is_cached = Column(Boolean, default=False)
    cache_hit_count = Column(Integer, default=0)            # Times this result reused
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="requests")

    __table_args__ = (
        Index("ix_requests_hash_endpoint", "request_hash", "endpoint"),
        Index("ix_requests_user_created", "user_id", "created_at"),
    )


# ─── Price Forecast ───────────────────────────────────────────────────────────

class PriceForecast(Base):
    __tablename__ = "forecasts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_hash = Column(String(64), nullable=False, index=True)
    crop = Column(String(50), nullable=False)
    forecast_period = Column(String(50), nullable=False)
    duration = Column(Integer, nullable=False)
    duration_type = Column(String(20), nullable=False)
    model_used = Column(String(50), nullable=False)
    subscription_tier = Column(String(20), nullable=False)
    predictions = Column(JSON, nullable=False)         # [{date, price, lower_ci, upper_ci}]
    accuracy_metrics = Column(JSON, nullable=False)    # {rmse, mape, forecast_bias}
    raw_result = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


# ─── Crop Suitability ─────────────────────────────────────────────────────────

class SuitabilityResult(Base):
    __tablename__ = "suitability_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_hash = Column(String(64), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    soil_ph = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    elevation = Column(Float, nullable=False)
    best_crops = Column(JSON, nullable=False) 
    region_estimate = Column(String(50), nullable=False)         # ["maize", "beans", ...]
    suitability_scores = Column(JSON, nullable=False)    # {"maize": 0.92, ...}
    recommended_crop = Column(String(50), nullable=False)
    model_used = Column(String(50), nullable=False)
    accuracy_metrics = Column(JSON, nullable=False)
    subscription_tier = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


# ─── Yield Prediction ─────────────────────────────────────────────────────────

class YieldPrediction(Base):
    __tablename__ = "yield_predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_hash = Column(String(64), nullable=False, index=True)
    crop = Column(String(50), nullable=False)
    land_size = Column(Float, nullable=False)
    soil_ph = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    elevation = Column(Float, nullable=False)
    irrigation = Column(Boolean, default=False)
    fertilizer_type = Column(String(50), nullable=True)
    predicted_yield_tons = Column(Float, nullable=False)
    yield_per_hectare = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(String(50), nullable=False)
    accuracy_metrics = Column(JSON, nullable=False)
    subscription_tier = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


# ─── Risk Score ───────────────────────────────────────────────────────────────

class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_hash = Column(String(64), nullable=False, index=True)
    crop = Column(String(50), nullable=False)
    region = Column(String(100), nullable=False)
    land_size = Column(Float, nullable=False)
    soil_ph = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    market_access = Column(String(50), nullable=False)
    overall_risk_score = Column(Float, nullable=False)
    risk_level = Column(Enum(RiskLevel), nullable=False)
    financial_risk = Column(Float, nullable=False)
    climate_risk = Column(Float, nullable=False)
    agronomic_risk = Column(Float, nullable=False)
    risk_factors = Column(JSON, nullable=True)           # Explanations
    recommendations = Column(JSON, nullable=True)
    accuracy_metrics = Column(JSON, nullable=False)
    subscription_tier = Column(String(20), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)


# ─── Reports ──────────────────────────────────────────────────────────────────

class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    request_id = Column(UUID(as_uuid=True), nullable=False)
    report_type = Column(String(50), nullable=False)   # price_forecast | suitability | yield | risk
    format = Column(Enum(ReportFormat), nullable=False)
    file_path = Column(String(500), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    is_ready = Column(Boolean, default=False)
    download_count = Column(Integer, default=0)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="reports")


# ─── Dataset Registry ─────────────────────────────────────────────────────────

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    crop = Column(String(50), nullable=True)
    category = Column(String(50), nullable=False)   # price | soil | climate | yield
    file_path = Column(String(500), nullable=False)
    row_count = Column(Integer, nullable=True)
    date_range_start = Column(DateTime, nullable=True)
    date_range_end = Column(DateTime, nullable=True)
    source = Column(String(200), nullable=True)
    last_updated = Column(DateTime(timezone=True), default=utcnow)
    created_at = Column(DateTime(timezone=True), default=utcnow)


# ─── Audit Log ────────────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)       # login | register | forecast | etc.
    resource = Column(String(100), nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    meta_data = Column("metadata", JSON, nullable=True)
    success = Column(Boolean, default=True)
    error_detail = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_logs_user_created", "user_id", "created_at"),
        Index("ix_audit_logs_action", "action"),
    )
