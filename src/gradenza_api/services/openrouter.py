"""
OpenRouter HTTP client with retry logic.

All model calls go through `call_openrouter()`.  Retries on 429 and
5xx responses with exponential backoff via tenacity.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from gradenza_api.settings import settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT = 120.0  # seconds — vision + reasoning models can be slow


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, _OpenRouterHTTPError):
        return exc.status_code in (429, 500, 502, 503, 504)
    return False


class _OpenRouterHTTPError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"OpenRouter {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def call_openrouter(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.1,
    timeout: float = _DEFAULT_TIMEOUT,
) -> str:
    """
    Call the OpenRouter chat completions endpoint.
    Returns the raw text content of the first choice.
    Raises _OpenRouterHTTPError on non-2xx responses (with retry on 429/5xx).
    """
    body = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )

    if not res.is_success:
        err_text = res.text
        logger.error(
            "OpenRouter error status=%s model=%s body_preview=%s",
            res.status_code,
            model,
            err_text[:300],
        )
        raise _OpenRouterHTTPError(res.status_code, err_text)

    data = res.json()
    content: str = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content


def strip_json_fences(text: str) -> str:
    """Remove optional ``` or ```json fences that some models still emit."""
    import re
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
