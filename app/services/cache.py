"""
Redis Caching Service
=====================
Manages result caching with TTL per subscription tier.
Provides request deduplication on top of DB-level deduplication.

Cache hierarchy:
  1. Redis (fast, ephemeral)  — checked first
  2. PostgreSQL (persistent)  — checked second
  3. Compute                  — only if both miss
"""

import json
from typing import Any, Optional

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import cache_logger


# ─── TTL per subscription tier (seconds) ─────────────────────────────────────
CACHE_TTL = {
    "free": 86400,       # 24 hours — free users always get cached
    "medium": 3600,      # 1 hour
    "premium": 900,      # 15 minutes — freshest data
}

_redis_client: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Return (or create) the shared async Redis connection."""
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def close_redis() -> None:
    """Gracefully close the Redis connection on shutdown."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


# ─── Core Cache Operations ────────────────────────────────────────────────────

async def cache_set(
    key: str,
    value: Any,
    tier: str = "free",
    ttl: Optional[int] = None,
) -> bool:
    """
    Store a value in Redis.

    Args:
        key: Cache key (usually the request hash)
        value: Serializable result dict/list
        tier: Subscription tier — determines TTL
        ttl: Override TTL in seconds

    Returns:
        True if set successfully
    """
    try:
        client = await get_redis()
        effective_ttl = ttl or CACHE_TTL.get(tier, settings.REDIS_CACHE_TTL)
        serialized = json.dumps(value, default=str)
        await client.setex(f"agri:{key}", effective_ttl, serialized)
        cache_logger.debug("Cache SET", key=key, ttl=effective_ttl)
        return True
    except Exception as exc:
        cache_logger.warning("Cache SET failed", key=key, error=str(exc))
        return False


async def cache_get(key: str) -> Optional[Any]:
    """
    Retrieve a cached value.

    Returns:
        Deserialized value, or None if not cached / expired
    """
    try:
        client = await get_redis()
        raw = await client.get(f"agri:{key}")
        if raw:
            cache_logger.debug("Cache HIT", key=key)
            return json.loads(raw)
        cache_logger.debug("Cache MISS", key=key)
        return None
    except Exception as exc:
        cache_logger.warning("Cache GET failed", key=key, error=str(exc))
        return None


async def cache_delete(key: str) -> bool:
    """Invalidate a specific cached result."""
    try:
        client = await get_redis()
        await client.delete(f"agri:{key}")
        return True
    except Exception:
        return False


async def cache_exists(key: str) -> bool:
    """Check if a key exists in cache without fetching the value."""
    try:
        client = await get_redis()
        return bool(await client.exists(f"agri:{key}"))
    except Exception:
        return False


# ─── Rate Limiting ────────────────────────────────────────────────────────────

async def increment_request_count(user_id: str, date_str: str) -> int:
    """
    Increment and return the daily request count for a user.
    Key expires at midnight (86400s TTL).

    Args:
        user_id: UUID string of the user
        date_str: Date string e.g. "2024-05-12"

    Returns:
        Current count after increment
    """
    try:
        client = await get_redis()
        key = f"rate:{user_id}:{date_str}"
        count = await client.incr(key)
        if count == 1:
            # First request today — set expiry for end of day
            await client.expire(key, 86400)
        return count
    except Exception as exc:
        cache_logger.error("Rate limit increment failed", user_id=user_id, error=str(exc))
        return 0


async def get_request_count(user_id: str, date_str: str) -> int:
    """Get current daily request count without incrementing."""
    try:
        client = await get_redis()
        key = f"rate:{user_id}:{date_str}"
        val = await client.get(key)
        return int(val) if val else 0
    except Exception:
        return 0


# ─── Cache Health ─────────────────────────────────────────────────────────────

async def ping_redis() -> bool:
    """Returns True if Redis is reachable."""
    try:
        client = await get_redis()
        return await client.ping()
    except Exception:
        return False
