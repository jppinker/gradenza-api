"""
POST /v1/canvas/chat

Canvas AI chat assistant for teachers.

Used by the Next.js proxy route /api/canvas/chat, which forwards the teacher's
Supabase JWT for auth + usage recording.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import call_openrouter
from gradenza_api.services.supabase_client import get_service_client
from gradenza_api.services.usage import AIUsage, record_ai_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/canvas", tags=["canvas"])

# ── Constants ──────────────────────────────────────────────────────────────────

CANVAS_CHAT_MODEL = "google/gemini-2.5-flash"
CANVAS_CHAT_TEMPERATURE = 0.2
MAX_PROMPT_LENGTH = 4000
MAX_HISTORY_TURNS = 12

SYSTEM_PROMPT = """\
You are a helpful teaching assistant embedded in a whiteboard canvas editor.

Write concise plain text suitable for placing directly onto a teaching canvas.
Prefer short paragraphs, bullet points, and clear steps.

When writing math, use KaTeX/LaTeX notation:
- inline math: $...$
- display math: $$...$$

Do not mention that you are an AI model. Do not add meta commentary.
"""


# ── Request / Response models ──────────────────────────────────────────────────

class CanvasChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CanvasChatRequest(BaseModel):
    canvas_id: str | None = None
    prompt: str = Field(..., min_length=1)
    conversation: list[CanvasChatTurn] = Field(default_factory=list)


class CanvasChatResponse(BaseModel):
    answer: str


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=CanvasChatResponse,
    summary="Chat with the Canvas AI assistant",
)
async def canvas_chat(
    body: CanvasChatRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher", "school_admin")),
    ],
) -> CanvasChatResponse:
    request_id = str(uuid.uuid4())[:8]

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt is required",
        )
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prompt too long (max {MAX_PROMPT_LENGTH} characters)",
        )

    logger.info(
        "[canvas-chat][%s] user=%s prompt_len=%d history_turns=%d canvas_id=%s",
        request_id,
        user.id,
        len(prompt),
        len(body.conversation),
        body.canvas_id,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in body.conversation[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": prompt})

    svc = get_service_client()
    caller_id = user.id or "internal"
    entity_id = body.canvas_id

    try:
        result = await call_openrouter(
            model=CANVAS_CHAT_MODEL,
            temperature=CANVAS_CHAT_TEMPERATURE,
            messages=messages,
        )
        answer = (result.content or "").strip()
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
    except Exception as exc:
        logger.error("[canvas-chat][%s] OpenRouter error: %s", request_id, exc)
        await record_ai_usage(
            svc,
            user_id=caller_id,
            source="canvas_chat",
            entity_type="canvas",
            entity_id=entity_id,
            route="POST /v1/canvas/chat",
            status="error",
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service unavailable",
        ) from exc

    await record_ai_usage(
        svc,
        user_id=caller_id,
        source="canvas_chat",
        entity_type="canvas",
        entity_id=entity_id,
        route="POST /v1/canvas/chat",
        usage=ai_usage,
        status="success",
    )

    logger.info(
        "[canvas-chat][%s] done answer_len=%d tokens=%s/%s",
        request_id,
        len(answer),
        ai_usage.prompt_tokens,
        ai_usage.completion_tokens,
    )

    return CanvasChatResponse(answer=answer)

