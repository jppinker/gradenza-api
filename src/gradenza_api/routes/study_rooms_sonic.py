"""
POST /v1/study-rooms/sonic

@sonic LLM bot for study room chats.
Called by the Next.js proxy at /api/study-rooms/sonic when a chat message
contains an @sonic mention. Returns the bot's answer; the Next.js route
inserts it into the DB as a bot message.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import call_openrouter
from gradenza_api.services.supabase_client import get_service_client
from gradenza_api.services.usage import AIUsage, record_ai_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/study-rooms", tags=["study-rooms"])

SONIC_MODEL = "google/gemini-2.5-flash"
SONIC_TEMPERATURE = 0.4
MAX_BODY_LENGTH = 3000

SYSTEM_PROMPT = """\
You are Sonic, a friendly and knowledgeable math study assistant living inside a Gradenza study room chat.

Your job is to help students understand math concepts, work through problems, and check their reasoning. \
You support IB Mathematics (Analysis & Approaches and Applications & Interpretation, both SL and HL), \
WAEC Mathematics, and general math topics.

Guidelines:
- Be concise and clear. Students want quick, useful answers, not essays.
- Use KaTeX/LaTeX notation for math: inline math with $...$, display math with $$...$$.
- Show working steps when solving problems.
- If a student's approach is wrong, gently correct it and explain why.
- If a question is ambiguous, answer the most likely interpretation.
- Do not invent questions or problems that were not asked.
- Do not mention that you are an AI model or reference your underlying technology.
- Address the student directly and keep a supportive, encouraging tone.
"""


class SonicMessageContext(BaseModel):
    body: str = Field(..., max_length=MAX_BODY_LENGTH)
    author_name: str | None = None


class SonicRequest(BaseModel):
    room_id: str
    channel_id: str
    trigger_message: SonicMessageContext
    reply_to_message: SonicMessageContext | None = None


class SonicResponse(BaseModel):
    answer: str


@router.post(
    "/sonic",
    response_model=SonicResponse,
    summary="@sonic LLM bot reply",
)
async def sonic_reply(
    body: SonicRequest,
    user: Annotated[AuthUser, Depends(require_roles("student", "teacher", "tutor", "co_teacher", "school_admin"))],
) -> SonicResponse:
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        "[sonic][%s] user=%s room=%s channel=%s",
        request_id,
        user.id,
        body.room_id,
        body.channel_id,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if body.reply_to_message:
        reply_author = body.reply_to_message.author_name or "a student"
        messages.append({
            "role": "user",
            "content": (
                f"[Context — message that was being replied to, sent by {reply_author}]\n"
                f"{body.reply_to_message.body}"
            ),
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I see the message being replied to.",
        })

    trigger_author = body.trigger_message.author_name or "a student"
    question = body.trigger_message.body
    messages.append({
        "role": "user",
        "content": f"{trigger_author} asked: {question}",
    })

    svc = get_service_client()

    try:
        result = await call_openrouter(
            model=SONIC_MODEL,
            temperature=SONIC_TEMPERATURE,
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
        logger.error("[sonic][%s] OpenRouter error: %s", request_id, exc)
        await record_ai_usage(
            svc,
            user_id=user.id or "internal",
            source="sonic_bot",
            entity_type="study_room_channel",
            entity_id=body.channel_id,
            route="POST /v1/study-rooms/sonic",
            status="error",
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service unavailable",
        ) from exc

    await record_ai_usage(
        svc,
        user_id=user.id or "internal",
        source="sonic_bot",
        entity_type="study_room_channel",
        entity_id=body.channel_id,
        route="POST /v1/study-rooms/sonic",
        usage=ai_usage,
        status="success",
    )

    logger.info(
        "[sonic][%s] done answer_len=%d tokens=%s/%s",
        request_id,
        len(answer),
        ai_usage.prompt_tokens,
        ai_usage.completion_tokens,
    )

    return SonicResponse(answer=answer)
