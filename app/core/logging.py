"""
Logging Module
==============
Structured JSON logging using structlog.
All log events include request_id, user_id, timestamps automatically.
"""

import logging
import sys
from typing import Any, Dict
import structlog
from app.core.config import settings


def setup_logging() -> None:
    """
    Configure structlog for structured JSON logging.
    Called once at application startup.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.LOG_FORMAT == "json":
        # Production: JSON output
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: colored console output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quieten noisy libraries
    for noisy in ["uvicorn.access", "httpx", "sqlalchemy.engine"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Return a named logger bound to the module."""
    return structlog.get_logger(name)


# ─── Application-level loggers ────────────────────────────────────────────────
logger = get_logger("agri_risk")
api_logger = get_logger("agri_risk.api")
ml_logger = get_logger("agri_risk.ml")
db_logger = get_logger("agri_risk.db")
cache_logger = get_logger("agri_risk.cache")
task_logger = get_logger("agri_risk.tasks")
