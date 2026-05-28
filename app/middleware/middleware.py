"""
Middleware Stack
================
Applied in order:
1. RequestLoggingMiddleware  — structured JSON logs for every request
2. RateLimitMiddleware       — per-user daily request limits by subscription
3. ErrorHandlingMiddleware   — catches unhandled exceptions
"""

import time
import uuid
from datetime import date
from typing import Callable

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logging import api_logger


# ─── Request Logging Middleware ───────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request with:
    - Unique request_id (UUID)
    - Method, path, status code
    - Processing duration in ms
    - Client IP
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # Bind request_id to all logs within this request context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Add request_id to response headers for client-side tracing
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            duration_ms = round((time.perf_counter() - start) * 1000, 2)

            api_logger.info(
                "HTTP request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                client_ip=request.client.host if request.client else "unknown",
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(duration_ms)
            return response

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            api_logger.error(
                "Unhandled request error",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise


# ─── Rate Limiting Middleware ─────────────────────────────────────────────────

RATE_LIMIT_PATHS = ["/api/v1/price-forecasting", "/api/v1/crop-suitability",
                    "/api/v1/yield-prediction", "/api/v1/risk-score"]

SUBSCRIPTION_LIMITS = {
    "free": settings.FREE_TIER_DAILY_LIMIT,
    "medium": settings.MEDIUM_TIER_DAILY_LIMIT,
    "premium": settings.PREMIUM_TIER_DAILY_LIMIT,
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Enforces daily request limits per user per subscription tier.
    Limits are tracked in Redis (key: rate:{user_id}:{date}).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only apply to ML analysis endpoints
        if not any(request.url.path.startswith(p) for p in RATE_LIMIT_PATHS):
            return await call_next(request)

        # Extract user from JWT (if present)
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return await call_next(request)

        try:
            from app.core.security import decode_token
            from app.services.cache import increment_request_count, get_request_count

            token = auth_header.split(" ")[1]
            payload = decode_token(token)
            user_id = payload.get("sub", "anonymous")
            subscription = payload.get("subscription_type", "free")
            daily_limit = SUBSCRIPTION_LIMITS.get(subscription, settings.FREE_TIER_DAILY_LIMIT)

            today = date.today().isoformat()
            current_count = await get_request_count(user_id, today)

            if current_count >= daily_limit:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Daily rate limit exceeded",
                        "detail": (
                            f"Your {subscription} plan allows {daily_limit} requests/day. "
                            f"Upgrade your subscription for higher limits."
                        ),
                        "daily_limit": daily_limit,
                        "current_count": current_count,
                        "reset_at": "midnight UTC",
                    },
                )

            # Increment counter
            await increment_request_count(user_id, today)

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(daily_limit)
            response.headers["X-RateLimit-Remaining"] = str(max(daily_limit - current_count - 1, 0))
            return response

        except Exception:
            # Don't block request if rate limiting fails
            return await call_next(request)


# ─── Error Handling Middleware ────────────────────────────────────────────────

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Catches all unhandled exceptions and returns a clean JSON error response.
    Prevents stack traces from leaking to clients in production.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            api_logger.exception(
                "Unhandled exception",
                path=request.url.path,
                error=str(exc),
                exc_info=exc,
            )

            if settings.DEBUG:
                detail = str(exc)
            else:
                detail = "An unexpected error occurred. Please try again."

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "detail": detail,
                    "status_code": 500,
                },
            )


# ─── Middleware Registration Helper ───────────────────────────────────────────

def register_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI app. Call at startup."""

    # CORS (must be outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handling (next outermost — catches all)
    app.add_middleware(ErrorHandlingMiddleware)

    # Rate limiting
    app.add_middleware(RateLimitMiddleware)

    # Request logging (innermost — logs after all processing)
    app.add_middleware(RequestLoggingMiddleware)
