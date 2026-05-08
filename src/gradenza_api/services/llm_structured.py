"""Helpers for robust structured JSON output from LLM calls.

Used by quiz builder blueprint + question generation to reduce failures from
intermittent malformed JSON output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, TypeVar

from pydantic import ValidationError

from gradenza_api.services.openrouter import (
    _OpenRouterHTTPError,
    OpenRouterResult,
    call_openrouter,
    strip_json_fences,
)

T = TypeVar("T")


def _preview_for_logs(text: str, *, limit: int = 800) -> str:
    if not text:
        return ""
    preview = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(preview) > limit:
        preview = preview[:limit] + "…"
    return preview


def _response_format_unsupported(exc: BaseException) -> bool:
    if not isinstance(exc, _OpenRouterHTTPError):
        return False
    if exc.status_code != 400:
        return False
    body = (exc.body or "").lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body


def _error_for_retry_prompt(exc: BaseException | None, *, limit: int = 400) -> str:
    if exc is None:
        return "unknown"
    msg = str(exc).replace("\r\n", "\n").replace("\r", "\n").strip()
    msg = " ".join(msg.split())
    if len(msg) > limit:
        msg = msg[:limit] + "…"
    return msg or exc.__class__.__name__


async def _call_openrouter_maybe_structured(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    response_format: dict[str, Any] | None,
) -> OpenRouterResult:
    try:
        return await call_openrouter(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
    except Exception as exc:
        # Some models/providers reject response_format; fall back to plain mode.
        if response_format is not None and _response_format_unsupported(exc):
            return await call_openrouter(model=model, messages=messages, temperature=temperature)
        raise


@dataclass
class StructuredAttempt:
    attempt: int
    llm_result: OpenRouterResult
    raw: str
    status: Literal["success", "json_parse_error", "validation_error"]
    error_message: str | None = None


@dataclass
class StructuredJSONResult(Generic[T]):
    value: T
    llm_result: OpenRouterResult
    attempt: int
    raw: str
    attempts: list[StructuredAttempt]


class StructuredJSONError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: list[StructuredAttempt],
        last_exc: BaseException | None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_exc = last_exc


async def call_openrouter_structured_json(
    *,
    model: str,
    base_messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None,
    temperature_first: float = 0.15,
    temperature_retry: float = 0.0,
    retry_user_instruction: str,
    validator: Callable[[Any], T],
    logger: logging.Logger,
    log_context: str,
) -> StructuredJSONResult[T]:
    """
    Two-attempt structured JSON helper:
    - Attempt 1: generate JSON with low temperature.
    - Attempt 2: include previous output and ask for a repair/fix with temperature 0.0.

    Retries only on JSON parse errors and validator failures.
    """
    last_raw = ""
    last_exc: BaseException | None = None
    attempts: list[StructuredAttempt] = []

    for attempt in (1, 2):
        temperature = temperature_first if attempt == 1 else temperature_retry
        messages = list(base_messages)
        if attempt == 2:
            retry_msg = f"{retry_user_instruction}\n\nPrevious parser/validation error: {_error_for_retry_prompt(last_exc)}"
            messages += [
                {"role": "assistant", "content": last_raw or "(no output)"},
                {"role": "user", "content": retry_msg},
            ]

        llm_result = await _call_openrouter_maybe_structured(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        last_raw = strip_json_fences(llm_result.content or "")

        try:
            parsed = json.loads(last_raw)
        except json.JSONDecodeError as exc:
            last_exc = exc
            attempts.append(
                StructuredAttempt(
                    attempt=attempt,
                    llm_result=llm_result,
                    raw=last_raw,
                    status="json_parse_error",
                    error_message=str(exc),
                )
            )
            logger.warning(
                "%s parse failed provider=openrouter model=%s attempt=%d err=%s preview=%s",
                log_context,
                model,
                attempt,
                exc,
                _preview_for_logs(last_raw),
            )
            continue

        try:
            validated = validator(parsed)
        except (ValidationError, ValueError, TypeError) as exc:
            last_exc = exc
            attempts.append(
                StructuredAttempt(
                    attempt=attempt,
                    llm_result=llm_result,
                    raw=last_raw,
                    status="validation_error",
                    error_message=str(exc),
                )
            )
            logger.warning(
                "%s validation failed provider=openrouter model=%s attempt=%d err=%s preview=%s",
                log_context,
                model,
                attempt,
                exc,
                _preview_for_logs(last_raw),
            )
            continue

        attempts.append(
            StructuredAttempt(
                attempt=attempt,
                llm_result=llm_result,
                raw=last_raw,
                status="success",
            )
        )
        return StructuredJSONResult(
            value=validated,
            llm_result=llm_result,
            attempt=attempt,
            raw=last_raw,
            attempts=attempts,
        )

    raise StructuredJSONError(
        f"{log_context} structured JSON failed after 2 attempts: {last_exc}",
        attempts=attempts,
        last_exc=last_exc,
    ) from last_exc
