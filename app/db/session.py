"""
Database Session Management
============================
Async SQLAlchemy engine, session factory, and FastAPI dependency.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import db_logger


# ─── Base ORM Class ───────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """All ORM models inherit from this base."""
    pass


# ─── Engine ───────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,           # Detect stale connections
    pool_recycle=3600,            # Recycle connections every hour
    echo=settings.DEBUG,          # Log SQL in debug mode
)


# ─── Session Factory ──────────────────────────────────────────────────────────

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,       # Prevent lazy-load after commit
    autocommit=False,
    autoflush=False,
)


# ─── FastAPI Dependency ───────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a scoped async database session.
    Automatically commits on success, rolls back on exception.

    Usage:
        @router.post("/something")
        async def handler(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            db_logger.error("DB session error — rolling back", error=str(exc))
            raise


# ─── Context Manager (for scripts/workers) ───────────────────────────────────

@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for use outside of FastAPI request lifecycle.
    Use in Celery workers and CLI scripts.

    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            db_logger.error("DB context error — rolling back", error=str(exc))
            raise


async def create_all_tables() -> None:
    """Create all tables defined in ORM models. Used for testing."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all_tables() -> None:
    """Drop all tables. Used for testing only."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
