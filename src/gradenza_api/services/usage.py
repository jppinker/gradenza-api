"""
Helpers for recording AI usage events to the ai_usage_events table.

All writes use the service-role Supabase client so RLS is bypassed.
Errors are logged and swallowed — usage recording must never crash the caller.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

logger = logging.getLogger(__name__)

# USD per million tokens: (prompt_price, completion_price)
_PRICING: dict[str, tuple[str, str]] = {
    "anthropic/claude-sonnet-4.6": ("3", "15"),
    "x-ai/grok-4.3": ("1.25", "2.5"),
    "google/gemini-3.1-flash-lite": ("0.25", "1.5"),
    "google/gemini-3.1-flash-lite-preview": ("0.25", "1.5"),
    "google/gemini-3.1-flash-image-preview": ("0.5", "3"),
    "google/gemini-3-pro-image-preview": ("2", "12"),
    "google/gemini-3-flash-preview": ("0.5", "3"),
    "google/gemini-2.5-flash-lite": ("0.1", "0.4"),
    "google/gemini-2.5-flash": ("0.3", "2.5"),
    "google/gemini-2.5-pro": ("1.25", "10"),
    "google/gemini-3.1-pro-preview": ("2", "12"),
}

_MILLION = Decimal("1000000")


@dataclass
class OpenRouterCost:
    cost_usd: float
    cost_credits: int


def calculate_openrouter_cost(
    model: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> OpenRouterCost | None:
    """
    Calculate cost from token counts using the OpenRouter pricing table.

    Returns None if the model is not in the pricing table.
    Missing token values default to 0.
    Uses Decimal arithmetic to avoid float rounding errors.
    """
    if not model or model not in _PRICING:
        return None

    prompt_price_str, completion_price_str = _PRICING[model]
    p_tokens = Decimal(prompt_tokens or 0)
    c_tokens = Decimal(completion_tokens or 0)
    raw_cost = (
        p_tokens * Decimal(prompt_price_str) / _MILLION
        + c_tokens * Decimal(completion_price_str) / _MILLION
    )

    cost_usd = float(raw_cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
    cost_credits = 0 if raw_cost == 0 else math.ceil(raw_cost * 100)
    return OpenRouterCost(cost_usd=cost_usd, cost_credits=cost_credits)


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
        cost = calculate_openrouter_cost(usage.model, usage.prompt_tokens, usage.completion_tokens)
        if cost is not None:
            row["cost_usd"] = cost.cost_usd
            row["cost_credits"] = cost.cost_credits
        elif usage.model:
            logger.warning(
                "record_ai_usage: no pricing for model=%s source=%s",
                usage.model,
                source,
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
