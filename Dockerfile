# ============================================================
# GEoN Risk Assessment Platform — Dockerfile
# Multi-stage build for minimal production image
# ============================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev libffi-dev build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Create required directories
RUN mkdir -p \
    ml_models/price ml_models/suitability ml_models/yield ml_models/risk \
    data/prices data/yields data/soil data/climate \
    reports/generated \
    && chmod -R 755 ml_models data reports

# Non-root user for security
RUN groupadd -r agri && useradd -r -g agri -d /app -s /bin/false agri
RUN chown -R agri:agri /app
USER agri

# Environment defaults (override via docker-compose or K8s secrets)
ENV APP_ENV=production \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Entrypoint
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-log", \
     "--log-level", "info"]
