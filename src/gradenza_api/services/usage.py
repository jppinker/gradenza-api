"""
Helpers for recording AI usage events to the ai_usage_events table.

All writes use the service-role Supabase client so RLS is bypassed.
Errors are logged and swallowed — usage recording must never crash the caller.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AIUsage:
    """Token counts and identifiers extracted from an OpenRouter response."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None
    request_id: str | None = None


async def record_ai_usage(
    svc_client: Any,
    *,
    user_id: str,
    source: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    route: str | None = None,
    usage: AIUsage | None = None,
    status: str = "success",
    error_message: str | None = None,
) -> None:
    """
    Insert one row into ai_usage_events.

    Never raises — any DB error is logged at ERROR level and swallowed so
    callers are not disrupted by a usage-recording failure.
    """
    row: dict[str, Any] = {
        "user_id": user_id,
        "source": source,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "route": route,
        "provider": "openrouter",
        "status": status,
        "error_message": error_message,
    }
    if usage is not None:
        row.update(
            {
                "model": usage.model,
                "request_id": usage.request_id,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        )

    def _insert() -> None:
        try:
            svc_client.table("ai_usage_events").insert(row).execute()
        except Exception as exc:
            logger.error(
                "record_ai_usage: DB insert failed source=%s user=%s: %s",
                source,
                user_id,
                exc,
            )

    await asyncio.to_thread(_insert)
