"""
API v1 Router
=============
Assembles all endpoint routers under /api/v1
"""

from fastapi import APIRouter

from app.api.v1.endpoints.auth import router as auth_router
from app.api.v1.endpoints.analysis import router as analysis_router
from app.api.v1.endpoints.reports import router as reports_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.me import router as history_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(analysis_router)
api_router.include_router(reports_router)
api_router.include_router(history_router)  # User history router from me.py
