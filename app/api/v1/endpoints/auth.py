"""
API Routes — Authentication
============================
POST /auth/register
POST /auth/login
POST /auth/refresh
GET  /auth/me
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user
)
from app.db.session import get_db
from app.models.models import User, SubscriptionType, AuditLog
from app.schemas.schemas import (
    UserRegisterRequest, UserLoginRequest, TokenResponse,
    RefreshTokenRequest, UserResponse
)
from app.core.logging import api_logger
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: UserRegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user. Default plan: FREE.
    Password is hashed with bcrypt before storage.
    """
    # Check email uniqueness
    result = await db.execute(select(User).where(User.email == payload.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        full_name=payload.full_name,
        email=payload.email,
        password_hash=hash_password(payload.password),
        subscription_type=SubscriptionType.free,
    )
    db.add(user)
    await db.flush()

    # Audit log
    db.add(AuditLog(
        user_id=user.id,
        action="register",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        success=True,
    ))

    await db.commit()
    await db.refresh(user)

    api_logger.info("New user registered", user_id=str(user.id), email=user.email)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    payload: UserLoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user. Returns JWT access + refresh tokens.
    """
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.password_hash):
        # Audit failed attempt
        if user:
            db.add(AuditLog(
                user_id=user.id, action="login_failed",
                ip_address=request.client.host if request.client else None,
                success=False, error_detail="Wrong password"
            ))
            await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account deactivated")

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.add(AuditLog(
        user_id=user.id, action="login",
        ip_address=request.client.host if request.client else None,
        success=True,
    ))
    await db.commit()

    access_token = create_access_token(
        str(user.id),
        extra_claims={"subscription_type": user.subscription_type.value, "email": user.email}
    )
    refresh_token = create_refresh_token(str(user.id))

    api_logger.info("User logged in", user_id=str(user.id))
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """Exchange a valid refresh token for a new access token."""
    token_data = decode_token(payload.refresh_token)
    if token_data.get("type") != "refresh":
        raise HTTPException(status_code=400, detail="Not a refresh token")

    result = await db.execute(select(User).where(User.id == token_data["sub"]))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    access_token = create_access_token(
        str(user.id),
        extra_claims={"subscription_type": user.subscription_type.value}
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=payload.refresh_token,  # Same refresh token
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return user
