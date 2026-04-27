"""
POST /v1/analytics/student-insight

LLM-powered chat endpoint: teacher asks questions about a student's
performance.  Context is assembled from Supabase based on the caller's
toggle selections (ContextOptions).
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import call_openrouter
from gradenza_api.services.student_context import assemble_context, verify_teacher_access
from gradenza_api.services.supabase_client import get_service_client
from gradenza_api.services.usage import AIUsage, record_ai_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/analytics", tags=["analytics"])

INSIGHT_MODEL = "google/gemini-2.5-flash"
INSIGHT_TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "You are an assistant for a teacher analyzing one student's performance. "
    "Answer concisely using ONLY the provided context. "
    "If a piece of information is not present in the context, say 'not in context' "
    "rather than making assumptions. "
    "Be specific: cite scores, topics, and dates when they are available."
)


# ── Request / Response models ──────────────────────────────────────────────────

class ContextOptions(BaseModel):
    overall_mastery: bool = True
    topic_mastery: bool = True
    recent_submissions: bool = True
    question_level_results: bool = False
    trust_layer_breakdown: bool = False
    attendance_and_lessons: bool = False
    teacher_notes: bool = False
    attempt_progression: bool = False
    window_days: int = Field(default=90, ge=1, le=365)


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class StudentInsightRequest(BaseModel):
    class_id: str
    student_id: str
    prompt: str = Field(..., min_length=1, max_length=2000)
    context_options: ContextOptions = Field(default_factory=ContextOptions)
    conversation: list[ChatTurn] = []


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class StudentInsightResponse(BaseModel):
    answer: str
    context_used: list[str]
    tokens: TokenUsage


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/student-insight",
    response_model=StudentInsightResponse,
    summary="Ask the LLM questions about a student's performance",
)
async def student_insight(
    body: StudentInsightRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher", "school_admin")),
    ],
) -> StudentInsightResponse:
    request_id = str(uuid.uuid4())[:8]

    logger.info(
        "[student-insight][%s] user=%s class=%s student=%s",
        request_id,
        user.id,
        body.class_id,
        body.student_id,
    )

    # Access check (skip for internal calls — no user.id to verify against)
    if not user.is_internal:
        allowed = await verify_teacher_access(
            class_id=body.class_id,
            student_id=body.student_id,
            teacher_id=user.id,  # type: ignore[arg-type]
            teacher_role=user.role,
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this student or class",
            )

    # Assemble context from enabled toggles
    context_md, context_used = await assemble_context(
        class_id=body.class_id,
        student_id=body.student_id,
        options=body.context_options,
    )

    logger.info(
        "[student-insight][%s] context sections=%s context_len=%d",
        request_id,
        context_used,
        len(context_md),
    )

    # Build message list: system + optional prior turns + new user message
    user_message = (
        f"## Student Performance Context\n\n{context_md}\n\n"
        f"---\n\n## Teacher's Question\n\n{body.prompt.strip()}"
    )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in body.conversation[-10:]:  # cap history at 10 turns
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_message})

    svc = get_service_client()
    caller_id = user.id or "internal"

    try:
        result = await call_openrouter(
            model=INSIGHT_MODEL,
            temperature=INSIGHT_TEMPERATURE,
            messages=messages,
        )
        answer = result.content
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
    except Exception as exc:
        logger.error("[student-insight][%s] OpenRouter error: %s", request_id, exc)
        await record_ai_usage(
            svc,
            user_id=caller_id,
            source="analytics",
            entity_type="student",
            entity_id=body.student_id,
            route="POST /v1/analytics/student-insight",
            status="error",
            error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM service error",
        ) from exc

    await record_ai_usage(
        svc,
        user_id=caller_id,
        source="analytics",
        entity_type="student",
        entity_id=body.student_id,
        route="POST /v1/analytics/student-insight",
        usage=ai_usage,
        status="success",
    )

    logger.info(
        "[student-insight][%s] done answer_len=%d tokens=%s/%s",
        request_id,
        len(answer),
        ai_usage.prompt_tokens,
        ai_usage.completion_tokens,
    )

    return StudentInsightResponse(
        answer=answer,
        context_used=context_used,
        tokens=TokenUsage(
            prompt_tokens=ai_usage.prompt_tokens,
            completion_tokens=ai_usage.completion_tokens,
        ),
    )
