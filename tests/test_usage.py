"""Unit tests for the OpenRouter pricing helper and record_ai_usage().

No database, network, or environment variables required.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gradenza_api.services.usage import (
    AIUsage,
    OpenRouterCost,
    calculate_openrouter_cost,
    record_ai_usage,
)


# ── calculate_openrouter_cost ──────────────────────────────────────────────────

def test_gemini_25_flash_known_values():
    """
    google/gemini-2.5-flash: prompt=0.3, completion=2.5 per million.

    raw_cost = 1675*0.3/1e6 + 556*2.5/1e6 = 0.0005025 + 0.00139 = 0.0018925
    cost_usd rounds to 0.001893 (6dp, 7th digit is 5 → round up)
    cost_credits = ceil(0.0018925 * 100) = ceil(0.18925) = 1
    """
    cost = calculate_openrouter_cost("google/gemini-2.5-flash", 1675, 556)
    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.001893, abs=1e-7)
    assert cost.cost_credits == 1


def test_claude_sonnet_higher_cost():
    """
    anthropic/claude-sonnet-4.6: prompt=3, completion=15 per million.

    raw_cost = 10000*3/1e6 + 5000*15/1e6 = 0.03 + 0.075 = 0.105
    cost_credits = ceil(0.105 * 100) = ceil(10.5) = 11
    """
    cost = calculate_openrouter_cost("anthropic/claude-sonnet-4.6", 10_000, 5_000)
    assert cost is not None
    assert cost.cost_usd == pytest.approx(0.105, abs=1e-7)
    assert cost.cost_credits == 11


def test_zero_tokens_returns_zero_cost():
    cost = calculate_openrouter_cost("google/gemini-2.5-flash", 0, 0)
    assert cost is not None
    assert cost.cost_usd == 0.0
    assert cost.cost_credits == 0


def test_none_tokens_treated_as_zero():
    cost = calculate_openrouter_cost("google/gemini-2.5-flash", None, None)
    assert cost is not None
    assert cost.cost_usd == 0.0
    assert cost.cost_credits == 0


def test_unknown_model_returns_none():
    cost = calculate_openrouter_cost("unknown/model-xyz", 1000, 500)
    assert cost is None


def test_none_model_returns_none():
    cost = calculate_openrouter_cost(None, 1000, 500)
    assert cost is None


def test_small_positive_cost_credits_at_least_one():
    """A small but representable cost (< 0.01 USD) must produce cost_credits >= 1.

    1000 prompt tokens at $0.1/M = raw_cost $0.0001 → cost_usd 0.000100, cost_credits = 1.
    """
    cost = calculate_openrouter_cost("google/gemini-2.5-flash-lite", 1000, 0)
    assert cost is not None
    assert cost.cost_usd > 0
    assert cost.cost_credits >= 1


def test_cost_usd_rounded_to_six_decimal_places():
    """cost_usd must be stored at 6dp precision."""
    cost = calculate_openrouter_cost("google/gemini-2.5-flash", 1675, 556)
    assert cost is not None
    # 6dp representation: 0.001893
    assert round(cost.cost_usd, 6) == cost.cost_usd


# ── record_ai_usage integration: cost fields inserted ─────────────────────────

async def test_record_ai_usage_inserts_cost_fields_for_known_model():
    """When model and tokens are known, the inserted row must include cost_usd and cost_credits."""
    mock_svc = MagicMock()
    mock_table = MagicMock()
    mock_svc.table.return_value = mock_table

    usage = AIUsage(
        model="google/gemini-2.5-flash",
        prompt_tokens=1675,
        completion_tokens=556,
        total_tokens=2231,
        request_id="req-abc",
    )

    import gradenza_api.services.usage as usage_module

    async def _to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    with patch.object(usage_module.asyncio, "to_thread", _to_thread):
        await record_ai_usage(
            mock_svc,
            user_id="user-1",
            source="quiz_builder",
            entity_type="quiz_session",
            entity_id="session-1",
            route="job:generate_quiz",
            usage=usage,
        )

    mock_svc.table.assert_called_once_with("ai_usage_events")
    insert_call = mock_table.insert.call_args
    row = insert_call.args[0]

    assert "cost_usd" in row
    assert "cost_credits" in row
    assert row["cost_usd"] == pytest.approx(0.001893, abs=1e-7)
    assert row["cost_credits"] == 1
    assert row["model"] == "google/gemini-2.5-flash"
    assert row["prompt_tokens"] == 1675
    assert row["completion_tokens"] == 556


async def test_record_ai_usage_no_cost_fields_for_unknown_model():
    """When the model is not in the pricing table, cost fields must be absent."""
    mock_svc = MagicMock()
    mock_table = MagicMock()
    mock_svc.table.return_value = mock_table

    usage = AIUsage(
        model="unknown/model-xyz",
        prompt_tokens=500,
        completion_tokens=100,
    )

    import gradenza_api.services.usage as usage_module

    async def _to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    with patch.object(usage_module.asyncio, "to_thread", _to_thread):
        await record_ai_usage(
            mock_svc,
            user_id="user-1",
            source="materials",
            usage=usage,
        )

    row = mock_table.insert.call_args.args[0]
    assert "cost_usd" not in row
    assert "cost_credits" not in row


async def test_record_ai_usage_no_cost_fields_when_usage_is_none():
    """Error rows with no usage metadata must not have cost fields."""
    mock_svc = MagicMock()
    mock_table = MagicMock()
    mock_svc.table.return_value = mock_table

    import gradenza_api.services.usage as usage_module

    async def _to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    with patch.object(usage_module.asyncio, "to_thread", _to_thread):
        await record_ai_usage(
            mock_svc,
            user_id="user-1",
            source="quiz_builder",
            status="error",
            error_message="timeout",
        )

    row = mock_table.insert.call_args.args[0]
    assert "cost_usd" not in row
    assert "cost_credits" not in row
    assert row["status"] == "error"


async def test_record_ai_usage_db_error_is_swallowed():
    """DB failures must be swallowed — record_ai_usage must not raise."""
    mock_svc = MagicMock()
    mock_table = MagicMock()
    mock_svc.table.return_value = mock_table
    mock_table.insert.return_value.execute.side_effect = Exception("connection refused")

    import gradenza_api.services.usage as usage_module

    async def _to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    with patch.object(usage_module.asyncio, "to_thread", _to_thread):
        # Should not raise
        await record_ai_usage(
            mock_svc,
            user_id="user-1",
            source="materials",
            usage=AIUsage(model="google/gemini-2.5-flash", prompt_tokens=100, completion_tokens=50),
        )
