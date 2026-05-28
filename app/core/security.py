"""
Security Module
===============
JWT token creation/validation, password hashing with bcrypt,
and authentication dependency injection for FastAPI.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db

# ─── Password Hashing ─────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

bearer_scheme = HTTPBearer()


def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    return pwd_context.verify(plain, hashed)


# ─── JWT Tokens ───────────────────────────────────────────────────────────────

def create_access_token(
    subject: str,
    extra_claims: Optional[Dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject: User ID (UUID as string)
        extra_claims: Additional payload fields (e.g. subscription_type)
        expires_delta: Custom expiry; defaults to ACCESS_TOKEN_EXPIRE_MINUTES

    Returns:
        Encoded JWT string
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    payload = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }

    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(subject: str) -> str:
    """Create a long-lived refresh token."""
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    payload = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.

    Raises:
        HTTPException 401 if token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── FastAPI Dependencies ─────────────────────────────────────────────────────

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """
    FastAPI dependency — extracts user ID from Bearer token.
    Use this in route handlers that require authentication.
    """
    payload = decode_token(credentials.credentials)
    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )
    return user_id


async def get_current_user(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    FastAPI dependency — returns the full User ORM object.
    Raises 404 if user no longer exists.
    """
    from app.models.models import User
    from sqlalchemy import select

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )
    return user


def require_subscription(*plans: str):
    """
    Dependency factory — restricts endpoint access by subscription plan.

    Usage:
        @router.post("/premium-endpoint")
        async def endpoint(user=Depends(require_subscription("premium"))):
            ...
    """
    async def _check(user=Depends(get_current_user)):
        if user.subscription_type not in plans:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This endpoint requires one of: {', '.join(plans)} plan(s). "
                       f"Your current plan: {user.subscription_type}",
            )
        return user
    return _check
