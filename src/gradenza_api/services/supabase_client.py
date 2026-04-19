"""
Supabase service-role client (sync).

The sync client is used throughout — wrapped in asyncio.to_thread()
wherever called from async contexts (route handlers, ARQ jobs).
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, TypeVar

from supabase import Client, create_client

from gradenza_api.settings import settings

_client: Client | None = None


def get_service_client() -> Client:
    """Return the module-level service-role Supabase client (lazy-initialised)."""
    global _client
    if _client is None:
        _client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _client


T = TypeVar("T")


async def run_sync(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous Supabase call in the default thread-pool executor."""
    return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))
