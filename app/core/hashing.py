"""
Request Hashing & Fingerprinting
=================================
Deterministic hashing of request parameters to enable cache deduplication.

CORE REQUIREMENT: If a request with identical parameters already exists,
return the cached result instead of recomputing.

Hash is SHA-256 of canonicalized request parameters, including:
- endpoint type
- crop name
- geographic parameters
- analysis parameters
- subscription tier (different tiers may produce different results)
"""

import hashlib
import json
from typing import Any, Dict, Optional


def _canonicalize(value: Any) -> Any:
    """
    Recursively convert a value to a canonical, deterministic form.
    Floats are rounded to 4 decimal places to avoid floating-point noise.
    """
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, str):
        return value.strip().lower()
    return value


def hash_request(
    endpoint: str,
    params: Dict[str, Any],
    subscription_tier: str = "free",
) -> str:
    """
    Generate a deterministic SHA-256 fingerprint for a request.

    Args:
        endpoint: API endpoint identifier (e.g. "price_forecast", "crop_suitability")
        params: All request parameters that affect the result
        subscription_tier: Plan tier — affects model quality, so included in hash

    Returns:
        64-character hex string (SHA-256)

    Example:
        hash_request(
            "price_forecast",
            {"crop": "Rice", "duration": 2, "duration_type": "months"},
            "premium"
        )
        # Always returns same hash for same inputs regardless of key order
    """
    canonical = {
        "endpoint": endpoint.strip().lower(),
        "params": _canonicalize(params),
        "tier": subscription_tier.strip().lower(),
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_price_forecast(
    crop: str,
    duration: int,
    duration_type: str,
    subscription_tier: str,
) -> str:
    return hash_request(
        "price_forecast",
        {"crop": crop, "duration": duration, "duration_type": duration_type},
        subscription_tier,
    )


def hash_crop_suitability(
    latitude: float,
    longitude: float,
    subscription_tier: str,
) -> str:
    return hash_request(
        "crop_suitability",
        {
            "lat": latitude,
            "lon": longitude,
        },
        subscription_tier,
    )


def hash_yield_prediction(
    crop: str,
    land_size: float,
    latitude: float,
    longitude: float,
    irrigation: bool,
    fertilizer: str,
    subscription_tier: str,
) -> str:
    return hash_request(
        "yield_prediction",
        {
            "crop": crop,
            "land_size": land_size,
            "latitude": latitude,
            "longitude": longitude,
            "irrigation": irrigation,
            "fertilizer": fertilizer,
        },
        subscription_tier,
    )


def hash_risk_score(
    crop: str,
    region: str,
    land_size: float,
    soil_ph: float,
    rainfall: float,
    temperature: float,
    market_access: str,
    subscription_tier: str,
) -> str:
    return hash_request(
        "risk_score",
        {
            "crop": crop,
            "region": region,
            "land_size": land_size,
            "soil_ph": soil_ph,
            "rainfall": rainfall,
            "temperature": temperature,
            "market_access": market_access,
        },
        subscription_tier,
    )
