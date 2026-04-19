"""
Auth helpers for the Gradenza API.

Two valid auth patterns accepted by endpoints:
  1. Supabase user JWT  (Authorization: Bearer <access_token>)
     → Verified via Supabase Auth API.  Returns AuthUser with id + role.
  2. Internal API secret (Authorization: Bearer <INTERNAL_API_SECRET>)
     → For server-to-server calls from the Next.js backend.
     → Returns AuthUser with id=None, role="internal".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gradenza_api.settings import settings

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


@dataclass
class AuthUser:
    id: str | None       # Supabase user UUID, or None for internal calls
    role: str            # "teacher", "tutor", "student", "co_teacher", "school_admin", "internal"
    is_internal: bool = False


async def _verify_supabase_token(token: str) -> AuthUser | None:
    """
    Call the Supabase Auth API to verify the user JWT.
    Returns AuthUser on success, None on failure.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            res = await client.get(
                f"{settings.supabase_url}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": settings.supabase_service_role_key,
                },
            )
        except httpx.RequestError as exc:
            logger.error("Supabase auth check network error: %s", exc)
            return None

        if res.status_code != 200:
            return None

        user_data = res.json()
        user_id: str = user_data.get("id", "")
        if not user_id:
            return None

    # Look up the user's role from public.users
    from gradenza_api.services.supabase_client import get_service_client
    import asyncio

    def _fetch_role() -> str:
        svc = get_service_client()
        result = (
            svc.table("users")
            .select("role")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        if result.data:
            return result.data.get("role", "teacher")
        return "teacher"

    role = await asyncio.to_thread(_fetch_role)
    return AuthUser(id=user_id, role=role)


async def get_auth_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> AuthUser:
    """
    FastAPI dependency: resolves the bearer token to an AuthUser.
    Raises 401 if the token is missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
        )

    token = credentials.credentials

    # Check internal secret first (fast path, no network call)
    if token == settings.internal_api_secret:
        return AuthUser(id=None, role="internal", is_internal=True)

    user = await _verify_supabase_token(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return user


def require_roles(*allowed_roles: str):
    """
    Returns a FastAPI dependency that enforces role restrictions.
    'internal' is always allowed (server-to-server).
    """

    async def _check(user: Annotated[AuthUser, Depends(get_auth_user)]) -> AuthUser:
        if user.is_internal:
            return user
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role}' is not permitted for this endpoint",
            )
        return user

    return _check
