"""
Online lesson AI endpoints:

  POST /v1/online-lesson/chat           — lesson planning conversation
  POST /v1/online-lesson/generate-plan  — generate a full lesson plan from chat context
  POST /v1/online-lesson/revise-plan    — revise an existing plan based on teacher feedback

Auth: lesson owners can mutate planning resources; co-teachers may read elsewhere.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import call_openrouter, strip_json_fences
from gradenza_api.services.supabase_client import get_service_client, run_sync
from gradenza_api.services.usage import AIUsage, record_ai_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/online-lesson", tags=["online-lessons"])

from gradenza_api.settings import settings  # noqa: E402 — imported after stdlib

# Model names are resolved from settings so they can be changed via env vars
# without touching code.  Defaults are in settings.py.
CHAT_TEMPERATURE = 0.6
MAX_HISTORY_TURNS = 16
MAX_MESSAGE_LENGTH = 3000

# ── JSON schema for structured AI response ─────────────────────────────────────

_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "readyToGenerateLessonPlan": {"type": "boolean"},
        "missingContext": {"type": "array", "items": {"type": "string"}},
        "detectedTopic": {"type": "string"},
        "detectedObjectives": {"type": "array", "items": {"type": "string"}},
        "suggestedQuestionCount": {"type": "integer"},
    },
    "required": [
        "reply",
        "readyToGenerateLessonPlan",
        "missingContext",
        "detectedTopic",
        "detectedObjectives",
        "suggestedQuestionCount",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are a lesson planning assistant embedded in an online lesson workspace for teachers.

Your job is to understand what the teacher wants to teach and help them feel ready to generate
a high-quality lesson plan — as quickly as possible.

You MUST respond with a JSON object in this exact format:
{
  "reply": "<conversational message to the teacher — 2–4 sentences max>",
  "readyToGenerateLessonPlan": <true or false>,
  "missingContext": ["<item 1>", "<item 2>"],
  "detectedTopic": "<main topic or empty string if unknown>",
  "detectedObjectives": ["<objective 1>", "<objective 2>"],
  "suggestedQuestionCount": <integer, or 0 if topic not yet clear>
}

Readiness rule — set readyToGenerateLessonPlan to TRUE when:
- A clear, specific lesson topic or teachable concept has been identified.
- The teacher's message contains enough intent to build a reasonable first draft.
- A vague single-word topic ("math", "calculus", "algebra") alone is NOT enough —
  there must be a specific concept, subtopic, or teachable goal
  (e.g. "trig graph transformations", "integration by parts", "solving simultaneous equations").
- Do NOT require student level, duration, exam system, or explicit objectives before enabling.

When readyToGenerateLessonPlan is TRUE:
- Tell the teacher they can generate the plan now.
- Mention that adding more context (level, duration, exam system) will make it tighter, but is optional.
- Do NOT ask another clarifying question unless something is genuinely ambiguous.
- missingContext lists OPTIONAL improvement suggestions (e.g. "Student level or exam system", "Lesson duration").

When readyToGenerateLessonPlan is FALSE (topic still vague or unknown):
- Ask exactly ONE focused question — tied directly to what the teacher mentioned, using their words.
  Good: teacher says "calculus" → "Which area of calculus — limits, derivatives, integrals, or something specific?"
  Good: teacher says "trig" → "Which aspect of trig — identities, solving equations, graphs, or transformations?"
  Good: teacher says "algebra" → "What part of algebra — factoring, simultaneous equations, or inequalities?"
  Bad: "Can you tell me more about what you'd like to teach?"
  Bad: "What are your learning goals for this lesson?"
  Bad: "Can you provide more details?"
- If the teacher mentioned any objectives or focus areas, acknowledge them before asking.
- Include examples inside the question so the teacher can reply with a single word or phrase.
- Do not ask about level, duration, or style until the topic is clear.
- missingContext lists what is actually missing to enable generation.

When readyToGenerateLessonPlan is TRUE and you offer an optional refinement:
- You may ask at most ONE specific, topic-tied question — only if it would genuinely improve the plan.
  Good (level unknown): "Should this be for IB AA HL, or do you want the plan to work for a general level first?"
  Good (focus unclear after a rich topic): "Do you want more emphasis on sketching graphs or on algebraic transformations?"
  Bad: "Is there anything else you'd like to include?"
  Bad: "What are your expectations for this lesson?"
- Always confirm the teacher can generate NOW before the optional question.
- If the teacher already provided level, focus, or duration — do not ask again.

Field rules:
- reply: warm, concise, action-oriented. Max 2–4 sentences.
- missingContext: up to 4 items — optional improvements when ready, actual blockers when not ready.
  Good item: "Student level or exam system (e.g. IB AA HL, GCSE, Grade 10 AP)"
  Good item: "Lesson duration"
  Good item: "Specific sub-topics to emphasize"
  Bad item: "More details" — never use this
  Bad item: "Learning objectives" when the teacher already stated objectives
- detectedTopic: best guess at the specific topic/concept, or "" if still unknown.
- detectedObjectives: extract specific learning outcomes from the teacher's messages; include objectives they stated even if implicitly (may be empty).
- suggestedQuestionCount: sensible practice question count for the scope (0 if topic unclear).

Avoid:
- Generic questions: "What are your goals?", "What are your expectations?", "Can you provide more details?"
- Asking multiple questions in one reply when the topic is already clear.
- Repeating questions the teacher already answered.
- Blocking generation because exam system, student level, duration, or style are missing.

Important:
- Do NOT invent information not provided by the teacher.
- Do NOT generate a full lesson plan in chat — that happens after the teacher clicks "Generate".
- Do NOT explain the JSON fields in your reply — just include the reply text naturally.
- Do NOT wrap JSON in markdown code fences.
"""


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatHistoryTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class LessonChatContext(BaseModel):
    students: list[dict[str, str]] = Field(default_factory=list)
    className: str | None = None
    existingPlan: str | None = None
    notesCount: int = 0


class OnlineLessonChatRequest(BaseModel):
    lesson_id: str
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    history: list[ChatHistoryTurn] = Field(default_factory=list)
    context: LessonChatContext | None = None


class OnlineLessonChatResponse(BaseModel):
    reply: str
    readyToGenerateLessonPlan: bool = False
    missingContext: list[str] = Field(default_factory=list)
    detectedTopic: str = ""
    detectedObjectives: list[str] = Field(default_factory=list)
    suggestedQuestionCount: int = 0


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def _verify_lesson_access(
    *,
    lesson_id: str,
    user: AuthUser,
    svc: Any,
    owner_only: bool = False,
) -> dict[str, Any]:
    """
    Fetch the lesson and verify the user can access it.
    When owner_only is true, only the lesson owner (or internal caller) can proceed.
    Returns the lesson row. Raises 403/404 on failure.
    """
    try:
        uuid.UUID(lesson_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid lesson_id",
        ) from exc

    def _fetch() -> dict | None:
        return (
            svc.table("lessons")
            .select(
                "id, teacher_id, kind, class_id, title, lesson_plan, "
                "lesson_plan_status, lesson_plan_generated_at, lesson_plan_approved_at, "
                "lesson_plan_source_meta, general_notes"
            )
            .eq("id", lesson_id)
            .maybe_single()
            .execute()
            .data
        )

    lesson = await run_sync(_fetch)
    if not lesson or lesson.get("kind") != "online":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lesson not found")

    if user.is_internal:
        return lesson

    if lesson.get("teacher_id") == user.id:
        return lesson

    if owner_only:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    class_id = lesson.get("class_id")
    if not class_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    def _fetch_co() -> dict | None:
        return (
            svc.table("class_co_teachers")
            .select("id")
            .eq("class_id", class_id)
            .eq("user_id", user.id)
            .not_.is_("accepted_at", "null")
            .maybe_single()
            .execute()
            .data
        )

    cot = await run_sync(_fetch_co)
    if not cot:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    return lesson


async def _fetch_lesson_context(*, lesson: dict[str, Any], svc: Any) -> dict[str, Any]:
    """
    Fetch students, class info, notes count, and materials count for the lesson.
    All supplementary — failures are swallowed and return empty data.
    """
    lesson_id = lesson["id"]
    class_id = lesson.get("class_id")

    async def _students() -> list[dict]:
        try:
            def _q() -> list:
                rows = (
                    svc.table("lesson_students")
                    .select("student_id, users!lesson_students_student_id_fkey(display_name, email)")
                    .eq("lesson_id", lesson_id)
                    .execute()
                    .data
                ) or []
                return rows
            rows = await run_sync(_q)
            return [
                {
                    "name": (r.get("users") or {}).get("display_name") or (r.get("users") or {}).get("email") or "Unknown",
                    "email": (r.get("users") or {}).get("email") or "",
                }
                for r in rows
            ]
        except Exception:
            return []

    async def _class_name() -> str | None:
        if not class_id:
            return None
        try:
            def _q() -> dict | None:
                return (
                    svc.table("classes")
                    .select("name")
                    .eq("id", class_id)
                    .maybe_single()
                    .execute()
                    .data
                )
            row = await run_sync(_q)
            return (row or {}).get("name")
        except Exception:
            return None

    async def _notes_count() -> int:
        try:
            def _q() -> int:
                result = (
                    svc.table("lesson_notes")
                    .select("id", count="exact")
                    .eq("lesson_id", lesson_id)
                    .execute()
                )
                return result.count or 0
            return await run_sync(_q)
        except Exception:
            return 0

    async def _materials_count() -> int:
        try:
            def _q() -> int:
                result = (
                    svc.table("lesson_materials")
                    .select("material_id", count="exact")
                    .eq("lesson_id", lesson_id)
                    .execute()
                )
                return result.count or 0
            return await run_sync(_q)
        except Exception:
            return 0

    students, class_name, notes_count, materials_count = await asyncio.gather(
        _students(), _class_name(), _notes_count(), _materials_count()
    )

    return {
        "students": students,
        "className": class_name,
        "existingPlan": lesson.get("lesson_plan") or "",
        "notesCount": notes_count,
        "materialsCount": materials_count,
    }


def _build_context_block(lesson: dict[str, Any], ctx: dict[str, Any]) -> str:
    """Format lesson + student context into a block appended to the system prompt."""
    parts: list[str] = [f"Lesson title: {lesson.get('title') or 'Untitled lesson'}"]

    if ctx.get("className"):
        parts.append(f"Class: {ctx['className']}")

    students = ctx.get("students") or []
    if students:
        names = ", ".join(s["name"] for s in students[:8])
        parts.append(f"Students ({len(students)}): {names}")

    existing_plan = (ctx.get("existingPlan") or "").strip()
    if existing_plan:
        preview = existing_plan[:400] + ("…" if len(existing_plan) > 400 else "")
        parts.append(f"Existing lesson plan (draft):\n{preview}")

    notes_count = ctx.get("notesCount") or 0
    if notes_count:
        parts.append(f"Existing student notes: {notes_count}")

    materials_count = ctx.get("materialsCount") or 0
    if materials_count:
        parts.append(f"Attached materials: {materials_count}")

    return "\n".join(parts)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _standard_artifact_meta(
    *,
    lesson: dict[str, Any],
    teacher_id: str | None,
    model: str | None,
    prompt_version: str,
    generated_at: str,
    approved_at: str | None = None,
    source_plan_hash: str | None = None,
    chat_history_turns: int | None = None,
) -> dict[str, Any]:
    return {
        "createdFrom": "online-lesson",
        "lessonId": lesson.get("id"),
        "lessonTitle": lesson.get("title") or "",
        "teacherId": teacher_id,
        "model": model,
        "promptVersion": prompt_version,
        "generatedAt": generated_at,
        "approvedAt": approved_at,
        "sourcePlanHash": source_plan_hash,
        "sourcePlanVersion": lesson.get("lesson_plan_generated_at"),
        "chatHistoryTurns": chat_history_turns,
    }


async def _fetch_recent_message_summary(*, svc: Any, lesson_id: str, limit: int = 8) -> tuple[int, list[str]]:
    try:
        def _q() -> tuple[int, list[dict]]:
            count_result = (
                svc.table("online_lesson_messages")
                .select("id", count="exact")
                .eq("lesson_id", lesson_id)
                .execute()
            )
            rows = (
                svc.table("online_lesson_messages")
                .select("role, content")
                .eq("lesson_id", lesson_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
                .data
            ) or []
            return count_result.count or 0, list(reversed(rows))

        count, rows = await run_sync(_q)
        summary = [
            f"{r.get('role')}: {str(r.get('content') or '').strip()[:240]}"
            for r in rows
            if str(r.get("content") or "").strip()
        ]
        return count, summary
    except Exception:
        return 0, []


async def _persist_messages(
    *,
    svc: Any,
    lesson_id: str,
    teacher_message: str,
    ai_reply: str,
    ai_metadata: dict[str, Any],
) -> None:
    """Insert teacher + AI messages into online_lesson_messages. Errors are swallowed."""
    def _insert() -> None:
        rows = [
            {
                "lesson_id": lesson_id,
                "role": "teacher",
                "content": teacher_message,
                "metadata": None,
            },
            {
                "lesson_id": lesson_id,
                "role": "ai",
                "content": ai_reply,
                "metadata": ai_metadata,
            },
        ]
        try:
            svc.table("online_lesson_messages").insert(rows).execute()
        except Exception as exc:
            logger.warning("[online-lesson-chat] persist failed: %s", exc)

    await asyncio.to_thread(_insert)


async def _call_with_structured_output(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Call OpenRouter requesting JSON output for the lesson chat response.
    Falls back from strict JSON schema → json_object → plain parse.
    """
    # Attempt 1: strict JSON schema
    model = settings.online_lesson_chat_model
    # Attempt 1: strict JSON schema
    try:
        result = await call_openrouter(
            model=model,
            messages=messages,
            temperature=CHAT_TEMPERATURE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "online_lesson_chat",
                    "strict": True,
                    "schema": _RESPONSE_JSON_SCHEMA,
                },
            },
        )
        return json.loads(strip_json_fences(result.content)), result
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("[online-lesson-chat] strict schema failed: %s", exc)

    # Attempt 2: json_object mode
    try:
        result = await call_openrouter(
            model=model,
            messages=messages,
            temperature=CHAT_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        return json.loads(strip_json_fences(result.content)), result
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("[online-lesson-chat] json_object mode failed: %s", exc)

    # Attempt 3: plain call
    result = await call_openrouter(
        model=model,
        messages=messages,
        temperature=CHAT_TEMPERATURE,
    )
    raw = strip_json_fences(result.content)
    return json.loads(raw), result  # propagate JSONDecodeError as 502


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=OnlineLessonChatResponse,
    summary="Lesson planning AI chat",
)
async def online_lesson_chat(
    body: OnlineLessonChatRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher")),
    ],
) -> OnlineLessonChatResponse:
    request_id = str(uuid.uuid4())[:8]
    svc = get_service_client()

    logger.info(
        "[online-lesson-chat][%s] user=%s lesson=%s history_turns=%d",
        request_id,
        user.id,
        body.lesson_id,
        len(body.history),
    )

    lesson = await _verify_lesson_access(lesson_id=body.lesson_id, user=user, svc=svc, owner_only=True)
    db_ctx = await _fetch_lesson_context(lesson=lesson, svc=svc)
    context_block = _build_context_block(lesson, db_ctx)

    system = SYSTEM_PROMPT
    if context_block.strip():
        system += f"\n\n## Current lesson context\n{context_block}"

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for turn in body.history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": body.message.strip()})

    ai_usage: AIUsage | None = None
    caller_id = user.id or "internal"
    route = "POST /v1/online-lesson/chat"

    try:
        parsed, result = await _call_with_structured_output(messages)
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
    except json.JSONDecodeError as exc:
        logger.error("[online-lesson-chat][%s] JSON parse failed: %s", request_id, exc)
        await record_ai_usage(
            svc, user_id=caller_id, source="online_lesson_chat",
            entity_type="lesson", entity_id=body.lesson_id,
            route=route, status="error", error_message=f"JSON parse: {exc}",
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service returned unexpected output — please try again",
        ) from exc
    except Exception as exc:
        logger.error("[online-lesson-chat][%s] OpenRouter error: %s", request_id, exc)
        await record_ai_usage(
            svc, user_id=caller_id, source="online_lesson_chat",
            entity_type="lesson", entity_id=body.lesson_id,
            route=route, status="error", error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service unavailable — please try again",
        ) from exc

    await record_ai_usage(
        svc, user_id=caller_id, source="online_lesson_chat",
        entity_type="lesson", entity_id=body.lesson_id,
        route=route, usage=ai_usage, status="success",
    )

    reply = (parsed.get("reply") or "").strip()
    if not reply:
        reply = "Sorry, I couldn't generate a response. Please try again."

    ready = bool(parsed.get("readyToGenerateLessonPlan", False))
    missing = [str(m) for m in (parsed.get("missingContext") or []) if m][:6]
    topic = str(parsed.get("detectedTopic") or "").strip()
    objectives = [str(o) for o in (parsed.get("detectedObjectives") or []) if o]
    q_count = int(parsed.get("suggestedQuestionCount") or 0)

    # Persistence is handled by the Next.js proxy route (app/api/online-lesson/chat/route.ts)
    # which has better error surfacing and auth context. Skip duplicate insert here.

    logger.info(
        "[online-lesson-chat][%s] done ready=%s topic=%r tokens=%s/%s",
        request_id,
        ready,
        topic,
        ai_usage.prompt_tokens,
        ai_usage.completion_tokens,
    )

    return OnlineLessonChatResponse(
        reply=reply,
        readyToGenerateLessonPlan=ready,
        missingContext=missing,
        detectedTopic=topic,
        detectedObjectives=objectives,
        suggestedQuestionCount=q_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Generate lesson plan
# ══════════════════════════════════════════════════════════════════════════════

GENERATE_PLAN_TEMPERATURE = 0.5
MAX_PLAN_HISTORY_TURNS = 20

_GENERATE_PLAN_SYSTEM = """\
You are an expert educational content designer helping a teacher create a detailed lesson plan.

You have been given the conversation the teacher had while planning this lesson.
Use ALL the context from that conversation to generate a thorough, teacher-facing lesson plan.

Format the plan as clean Markdown. Use ## for top-level sections, **bold** for terms,
and KaTeX-compatible math where needed: $inline$ and $$block$$.

The plan MUST contain these sections in this order, each as a ## heading:
1. Lesson Overview  (topic, target students, duration if mentioned)
2. Learning Objectives  (bulleted list, 3–5 specific outcomes)
3. Prerequisite Knowledge  (what students must already know)
4. Theory & Explanation  (core content, written for the teacher to present)
5. Key Definitions  (a definition list of the most important terms)
6. Key Concepts  (short bullets on what to emphasise)
7. Worked Examples  (2–4 step-by-step worked examples with full workings)
8. Common Misconceptions  (bullet list with explanation of each)
9. Suggested Pacing  (time allocation per activity, totalling the lesson duration)
10. What to Practice Next  (recommended follow-up exercises or topics)

Rules:
- Write for the teacher, not the student.
- Be specific and exam-aware if an exam system (IB, SAT, etc.) was mentioned.
- Do not add extra sections beyond the 10 listed above.
- Do not include a preamble or postamble — start directly with ## Lesson Overview.
- Return only the Markdown. No code fences around it.
"""


class GeneratePlanRequest(BaseModel):
    lesson_id: str
    chat_history: list[ChatHistoryTurn] = Field(default_factory=list)


class GeneratePlanResponse(BaseModel):
    plan: str
    source_meta: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/generate-plan",
    response_model=GeneratePlanResponse,
    summary="Generate a structured lesson plan from planning chat context",
)
async def generate_lesson_plan(
    body: GeneratePlanRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher")),
    ],
) -> GeneratePlanResponse:
    request_id = str(uuid.uuid4())[:8]
    svc = get_service_client()

    logger.info(
        "[generate-plan][%s] user=%s lesson=%s history_turns=%d",
        request_id, user.id, body.lesson_id, len(body.chat_history),
    )

    lesson = await _verify_lesson_access(lesson_id=body.lesson_id, user=user, svc=svc, owner_only=True)
    db_ctx = await _fetch_lesson_context(lesson=lesson, svc=svc)
    context_block = _build_context_block(lesson, db_ctx)

    system = _GENERATE_PLAN_SYSTEM
    if context_block.strip():
        system += f"\n\n## Lesson context\n{context_block}"

    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    history = body.chat_history[-MAX_PLAN_HISTORY_TURNS:]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})

    if not history:
        messages.append({
            "role": "user",
            "content": "Please generate the lesson plan based on the lesson context above.",
        })
    else:
        messages.append({
            "role": "user",
            "content": (
                "Based on everything we've discussed, please generate the complete lesson plan now."
            ),
        })

    caller_id = user.id or "internal"
    route = "POST /v1/online-lesson/generate-plan"

    try:
        result = await call_openrouter(
            model=settings.online_lesson_plan_model,
            messages=messages,
            temperature=GENERATE_PLAN_TEMPERATURE,
        )
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
    except Exception as exc:
        logger.error("[generate-plan][%s] OpenRouter error: %s", request_id, exc)
        await record_ai_usage(
            svc, user_id=caller_id, source="online_lesson_generate_plan",
            entity_type="lesson", entity_id=body.lesson_id,
            route=route, status="error", error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service unavailable — please try again",
        ) from exc

    await record_ai_usage(
        svc, user_id=caller_id, source="online_lesson_generate_plan",
        entity_type="lesson", entity_id=body.lesson_id,
        route=route, usage=ai_usage, status="success",
    )

    plan_text = (result.content or "").strip()
    if not plan_text:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned an empty plan — please try again",
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    source_meta = {
        **_standard_artifact_meta(
            lesson=lesson,
            teacher_id=user.id,
            model=result.usage.model,
            prompt_version="online_lesson_plan_v1",
            generated_at=generated_at,
            source_plan_hash=_hash_text(plan_text),
            chat_history_turns=len(history),
        ),
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "studentCount": len(db_ctx.get("students") or []),
        "className": db_ctx.get("className"),
    }

    logger.info(
        "[generate-plan][%s] done plan_len=%d tokens=%s/%s",
        request_id, len(plan_text),
        ai_usage.prompt_tokens, ai_usage.completion_tokens,
    )

    return GeneratePlanResponse(plan=plan_text, source_meta=source_meta)


# ══════════════════════════════════════════════════════════════════════════════
# Revise lesson plan
# ══════════════════════════════════════════════════════════════════════════════

REVISE_PLAN_TEMPERATURE = 0.4
MAX_REVISION_REQUEST_LENGTH = 1000

_REVISE_PLAN_SYSTEM = """\
You are an expert educational content designer helping a teacher revise an existing lesson plan.

The teacher will provide:
1. Their current lesson plan (in Markdown)
2. Specific feedback on what to change or improve

Your job: return the COMPLETE revised lesson plan incorporating the teacher's feedback.

Rules:
- Preserve the exact same 10-section Markdown structure as the original.
- Only change what the teacher asked to change — keep everything else.
- Return only the Markdown. No preamble, no postamble, no code fences.
"""


class RevisePlanRequest(BaseModel):
    lesson_id: str
    current_plan: str = Field(..., min_length=10)
    revision_request: str = Field(..., min_length=1, max_length=MAX_REVISION_REQUEST_LENGTH)


class RevisePlanResponse(BaseModel):
    plan: str


@router.post(
    "/revise-plan",
    response_model=RevisePlanResponse,
    summary="Revise an existing lesson plan based on teacher feedback",
)
async def revise_lesson_plan(
    body: RevisePlanRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher")),
    ],
) -> RevisePlanResponse:
    request_id = str(uuid.uuid4())[:8]
    svc = get_service_client()

    logger.info(
        "[revise-plan][%s] user=%s lesson=%s",
        request_id, user.id, body.lesson_id,
    )

    await _verify_lesson_access(lesson_id=body.lesson_id, user=user, svc=svc, owner_only=True)

    user_message = (
        f"Here is my current lesson plan:\n\n{body.current_plan}\n\n"
        f"---\n\nPlease make the following changes:\n{body.revision_request.strip()}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _REVISE_PLAN_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    caller_id = user.id or "internal"
    route = "POST /v1/online-lesson/revise-plan"

    try:
        result = await call_openrouter(
            model=settings.online_lesson_revise_model,
            messages=messages,
            temperature=REVISE_PLAN_TEMPERATURE,
        )
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
    except Exception as exc:
        logger.error("[revise-plan][%s] OpenRouter error: %s", request_id, exc)
        await record_ai_usage(
            svc, user_id=caller_id, source="online_lesson_revise_plan",
            entity_type="lesson", entity_id=body.lesson_id,
            route=route, status="error", error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service unavailable — please try again",
        ) from exc

    await record_ai_usage(
        svc, user_id=caller_id, source="online_lesson_revise_plan",
        entity_type="lesson", entity_id=body.lesson_id,
        route=route, usage=ai_usage, status="success",
    )

    plan_text = (result.content or "").strip()
    if not plan_text:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned an empty revision — please try again",
        )

    logger.info(
        "[revise-plan][%s] done plan_len=%d tokens=%s/%s",
        request_id, len(plan_text),
        ai_usage.prompt_tokens, ai_usage.completion_tokens,
    )

    return RevisePlanResponse(plan=plan_text)


# ══════════════════════════════════════════════════════════════════════════════
# Generate practice questions
# ══════════════════════════════════════════════════════════════════════════════

GENERATE_QUESTIONS_TEMPERATURE = 0.5

# ── Generation options Pydantic models ────────────────────────────────────────

class TypeMix(BaseModel):
    multiple_choice: int = Field(default=0, ge=0, le=20)
    short_answer: int = Field(default=0, ge=0, le=20)
    long_answer: int = Field(default=0, ge=0, le=20)
    fill_blank: int = Field(default=0, ge=0, le=20)

    @model_validator(mode="before")
    @classmethod
    def normalize_labels(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        aliases = {
            "mcq": "multiple_choice",
            "mc": "multiple_choice",
            "multipleChoice": "multiple_choice",
            "multiple_choice": "multiple_choice",
            "short": "short_answer",
            "shortAnswer": "short_answer",
            "short_answer": "short_answer",
            "frq": "long_answer",
            "free_response": "long_answer",
            "long": "long_answer",
            "longAnswer": "long_answer",
            "long_answer": "long_answer",
            "fill": "fill_blank",
            "fillBlank": "fill_blank",
            "fill_blank": "fill_blank",
        }
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            canonical = aliases.get(str(key), str(key))
            normalized[canonical] = value
        return normalized


class DifficultyMix(BaseModel):
    easy: int = Field(default=0, ge=0, le=20)
    medium: int = Field(default=0, ge=0, le=20)
    challenge: int = Field(default=0, ge=0, le=20)

    @model_validator(mode="before")
    @classmethod
    def normalize_labels(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        aliases = {
            "easy": "easy",
            "foundation": "easy",
            "medium": "medium",
            "standard": "medium",
            "challenge": "challenge",
            "challenging": "challenge",
            "hard": "challenge",
        }
        normalized: dict[str, Any] = {}
        for key, value in data.items():
            canonical = aliases.get(str(key), str(key))
            normalized[canonical] = value
        return normalized


_DEFAULT_DIFFICULTY_DEFS: dict[str, str] = {
    "easy": "Straightforward recall or basic comprehension; suitable for checking core understanding.",
    "medium": "Requires explanation, comparison, or applying the content in a familiar context.",
    "challenge": "Requires deeper reasoning, synthesis, evaluation, or applying the content in a less familiar context.",
}


class DifficultyDefinitions(BaseModel):
    easy: str = Field(default=_DEFAULT_DIFFICULTY_DEFS["easy"], max_length=500)
    medium: str = Field(default=_DEFAULT_DIFFICULTY_DEFS["medium"], max_length=500)
    challenge: str = Field(default=_DEFAULT_DIFFICULTY_DEFS["challenge"], max_length=500)


class GenerationOptions(BaseModel):
    question_count: int | None = Field(default=None, ge=1, le=20)
    type_mix: TypeMix | None = None
    difficulty_mix: DifficultyMix | None = None
    difficulty_definitions: DifficultyDefinitions = Field(default_factory=DifficultyDefinitions)
    marks_min: int | None = Field(default=None, ge=1, le=20)
    marks_max: int | None = Field(default=None, ge=1, le=20)
    total_marks_target: int | None = Field(default=None, ge=1, le=200)
    exam_style: str | None = Field(default=None, max_length=100)
    include_worked_solutions: bool = True
    include_markscheme: bool = True
    include_tags: bool = False
    teacher_instructions: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_marks_range(self) -> "GenerationOptions":
        if self.marks_min is not None and self.marks_max is not None:
            if self.marks_min > self.marks_max:
                raise ValueError(
                    f"marks_min ({self.marks_min}) must be ≤ marks_max ({self.marks_max})"
                )
        return self


def _build_options_block(opts: GenerationOptions, question_count: int) -> str:
    """Return a prompt block describing generation requirements from options."""
    lines: list[str] = []

    tm = opts.type_mix
    if tm:
        specified = [
            f"{tm.multiple_choice} multiple_choice" if tm.multiple_choice else None,
            f"{tm.short_answer} short_answer" if tm.short_answer else None,
            f"{tm.long_answer} long_answer" if tm.long_answer else None,
            f"{tm.fill_blank} fill_blank" if tm.fill_blank else None,
        ]
        specified = [s for s in specified if s]
        if specified:
            total = tm.multiple_choice + tm.short_answer + tm.long_answer + tm.fill_blank
            remaining = max(0, question_count - total)
            lines.append(f"Type distribution: {', '.join(specified)}"
                         + (f" — fill remaining {remaining} as short_answer" if remaining > 0 else ""))

    dm = opts.difficulty_mix
    if dm:
        specified = [
            f"{dm.easy} easy" if dm.easy else None,
            f"{dm.medium} medium" if dm.medium else None,
            f"{dm.challenge} challenge" if dm.challenge else None,
        ]
        specified = [s for s in specified if s]
        if specified:
            total = dm.easy + dm.medium + dm.challenge
            remaining = max(0, question_count - total)
            lines.append(f"Difficulty distribution: {', '.join(specified)}"
                         + (f" — fill remaining {remaining} as medium" if remaining > 0 else ""))

    defs = opts.difficulty_definitions or DifficultyDefinitions()
    lines.append(
        "Difficulty definitions (apply exactly when assigning and writing each level):\n"
        f"- Easy: {defs.easy.strip() or _DEFAULT_DIFFICULTY_DEFS['easy']}\n"
        f"- Medium: {defs.medium.strip() or _DEFAULT_DIFFICULTY_DEFS['medium']}\n"
        f"- Challenge: {defs.challenge.strip() or _DEFAULT_DIFFICULTY_DEFS['challenge']}"
    )

    if opts.marks_min is not None and opts.marks_max is not None:
        lines.append(f"Marks per question: {opts.marks_min}–{opts.marks_max}")
    elif opts.marks_min is not None:
        lines.append(f"Minimum marks per question: {opts.marks_min}")
    elif opts.marks_max is not None:
        lines.append(f"Maximum marks per question: {opts.marks_max}")
    if opts.total_marks_target is not None:
        lines.append(f"Total marks target: {opts.total_marks_target} — distribute evenly")

    if opts.exam_style:
        lines.append(f"Exam style / curriculum: {opts.exam_style} — phrase questions, command words, and mark allocation to match this style")

    if opts.include_worked_solutions:
        lines.append("'worked_solution' field: full step-by-step worked solution — not just the final answer")
    else:
        lines.append("'worked_solution' field: brief 1-line answer only")

    if opts.include_markscheme:
        lines.append("'markscheme_steps': include detailed mark-allocation guidance per step (e.g. 'M1 correct method, A1 answer')")

    if opts.include_tags:
        lines.append("'tags' field: include 3-5 relevant keyword tags per question")

    if opts.teacher_instructions:
        lines.append(f"\nTeacher instructions (follow these carefully):\n{opts.teacher_instructions}")

    return "\n".join(lines)


_SUGGEST_OPTIONS_SYSTEM = """\
You suggest compact defaults for generating practice questions from an approved lesson plan.

Return a single JSON object with exactly this shape:
{
  "generation_options": {
    "question_count": 5,
    "type_mix": {"multiple_choice": 1, "short_answer": 3, "long_answer": 1, "fill_blank": 0},
    "difficulty_mix": {"easy": 1, "medium": 3, "challenge": 1},
    "marks_min": 1,
    "marks_max": 6,
    "total_marks_target": 15,
    "exam_style": "IB AA HL",
    "include_worked_solutions": true,
    "include_markscheme": true,
    "include_tags": true,
    "teacher_instructions": "Short instruction that captures the lesson focus."
  },
  "rationale": "One short sentence explaining why these settings fit."
}

Rules:
- question_count must be 1-20.
- Type mix and difficulty mix totals must not exceed question_count.
- Use the lesson plan objectives and exam/curriculum context if present.
- Prefer 5 questions unless the plan is very narrow or broad.
- Keep teacher_instructions under 280 characters.
- Do not include student names or private notes in the output.
- Return only JSON. No markdown fences.
"""

_SUGGEST_OPTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["generation_options", "rationale"],
    "properties": {
        "generation_options": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "question_count", "type_mix", "difficulty_mix", "marks_min",
                "marks_max", "total_marks_target", "exam_style",
                "include_worked_solutions", "include_markscheme", "include_tags",
                "teacher_instructions",
            ],
            "properties": {
                "question_count": {"type": "integer"},
                "type_mix": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["multiple_choice", "short_answer", "long_answer", "fill_blank"],
                    "properties": {
                        "multiple_choice": {"type": "integer"},
                        "short_answer": {"type": "integer"},
                        "long_answer": {"type": "integer"},
                        "fill_blank": {"type": "integer"},
                    },
                },
                "difficulty_mix": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["easy", "medium", "challenge"],
                    "properties": {
                        "easy": {"type": "integer"},
                        "medium": {"type": "integer"},
                        "challenge": {"type": "integer"},
                    },
                },
                "marks_min": {"type": "integer"},
                "marks_max": {"type": "integer"},
                "total_marks_target": {"type": "integer"},
                "exam_style": {"type": "string"},
                "include_worked_solutions": {"type": "boolean"},
                "include_markscheme": {"type": "boolean"},
                "include_tags": {"type": "boolean"},
                "teacher_instructions": {"type": "string"},
            },
        },
        "rationale": {"type": "string"},
    },
}


class SuggestQuestionOptionsRequest(BaseModel):
    lesson_id: str


class SuggestQuestionOptionsResponse(BaseModel):
    generation_options: GenerationOptions
    rationale: str = ""


def _default_question_options() -> GenerationOptions:
    return GenerationOptions(
        question_count=5,
        type_mix=TypeMix(multiple_choice=1, short_answer=3, long_answer=1, fill_blank=0),
        difficulty_mix=DifficultyMix(easy=1, medium=3, challenge=1),
        difficulty_definitions=DifficultyDefinitions(),
        marks_min=1,
        marks_max=6,
        total_marks_target=15,
        exam_style="",
        include_worked_solutions=True,
        include_markscheme=True,
        include_tags=True,
        teacher_instructions="",
    )


@router.post(
    "/suggest-question-options",
    response_model=SuggestQuestionOptionsResponse,
    summary="Suggest practice question generation options from an approved lesson plan",
)
async def suggest_question_options(
    body: SuggestQuestionOptionsRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher")),
    ],
) -> SuggestQuestionOptionsResponse:
    request_id = str(uuid.uuid4())[:8]
    svc = get_service_client()
    lesson = await _verify_lesson_access(lesson_id=body.lesson_id, user=user, svc=svc, owner_only=True)

    lesson_plan = (lesson.get("lesson_plan") or "").strip()
    if not lesson_plan:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Lesson plan is empty")
    if lesson.get("lesson_plan_status") != "approved":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Approve the lesson plan before suggesting practice question options",
        )

    db_ctx = await _fetch_lesson_context(lesson=lesson, svc=svc)
    message_count, message_summary = await _fetch_recent_message_summary(svc=svc, lesson_id=body.lesson_id)

    context = [
        f"Lesson title: {lesson.get('title') or 'Untitled'}",
        f"Class: {db_ctx.get('className') or 'Not specified'}",
        f"Student count: {len(db_ctx.get('students') or [])}",
        f"Chat history turns: {message_count}",
    ]
    if message_summary:
        context.append("Recent planning chat summary:\n" + "\n".join(message_summary))

    messages = [
        {"role": "system", "content": _SUGGEST_OPTIONS_SYSTEM},
        {
            "role": "user",
            "content": "\n".join(context) + f"\n\nApproved lesson plan:\n{lesson_plan[:6000]}",
        },
    ]

    caller_id = user.id or "internal"
    route = "POST /v1/online-lesson/suggest-question-options"
    try:
        result = await call_openrouter(
            model=settings.online_lesson_chat_model,
            messages=messages,
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "practice_question_options",
                    "strict": True,
                    "schema": _SUGGEST_OPTIONS_SCHEMA,
                },
            },
        )
        parsed = json.loads(strip_json_fences(result.content or "{}"))
        opts_raw = parsed.get("generation_options") if isinstance(parsed, dict) else None
        opts = GenerationOptions.model_validate(opts_raw or {})
        count = opts.question_count or 5
        GenerateQuestionsRequest(
            lesson_id=body.lesson_id,
            question_count=count,
            generation_options=opts,
        )
        await record_ai_usage(
            svc,
            user_id=caller_id,
            source="online_lesson_suggest_question_options",
            entity_type="lesson",
            entity_id=body.lesson_id,
            route=route,
            usage=AIUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                model=result.usage.model,
                request_id=result.usage.request_id,
            ),
            status="success",
        )
        return SuggestQuestionOptionsResponse(
            generation_options=opts,
            rationale=str(parsed.get("rationale") or "").strip() if isinstance(parsed, dict) else "",
        )
    except Exception as exc:
        logger.warning("[suggest-question-options][%s] failed: %s", request_id, exc)
        await record_ai_usage(
            svc,
            user_id=caller_id,
            source="online_lesson_suggest_question_options",
            entity_type="lesson",
            entity_id=body.lesson_id,
            route=route,
            status="error",
            error_message=str(exc),
        )
        # Frontend treats this as non-blocking and falls back to local defaults.
        return SuggestQuestionOptionsResponse(
            generation_options=_default_question_options(),
            rationale="Using default practice settings because suggestions were unavailable.",
        )


_GENERATE_QUESTIONS_SYSTEM = """\
You are an expert assessment designer. Generate high-quality, teacher-ready practice questions from an approved lesson plan.

Your output MUST be a single JSON object: {"questions": [<question objects>]}

Each question object must contain ALL of these fields with the correct types:

{
  "slot": <integer — question number, starting at 1>,
  "title": "<concise title describing what the question tests — 4–10 words>",
  "question_type": "multiple_choice" | "short_answer" | "long_answer" | "fill_blank",
  "difficulty": "easy" | "medium" | "challenge",
  "marks": <integer 1–10>,
  "estimated_time_minutes": <integer 1–15>,
  "question_text": "<complete, self-contained question — include all information needed to answer>",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."] for multiple_choice, [] for all other types,
  "correct_answer": "<for MCQ: the letter only, e.g. 'B'; for other types: full model answer>",
  "worked_solution": "<complete step-by-step worked solution useful for both teacher and student>",
  "markscheme_steps": [
    {"description": "<what earns this mark — specific and actionable>", "marks": <integer>},
    ...
  ],
  "common_mistakes": ["<concrete mistake 1 students actually make on this topic>", "<mistake 2>"],
  "hints": ["<hint 1 that guides without giving the answer>"],
  "tags": ["<tag 1>", "<tag 2>", "<tag 3>"],
  "domain": "<curriculum domain or unit, e.g. Trigonometry, Mechanics>",
  "topic": "<specific topic, e.g. Amplitude and Period>",
  "subtopic": "<subtopic or focus area, or empty string if none>",
  "exam_system": "<exam system if known, e.g. IB, GCSE, SAT — or empty string>",
  "subject": "<subject, e.g. Mathematics, Physics — or empty string>",
  "level": "<level/tier, e.g. HL, SL, Standard, Advanced — or empty string>",
  "source_lesson_plan_excerpt": "<15–30 verbatim words from the lesson plan that this question tests>",
  "quality_notes": "<1-sentence note on why this question is good or what to watch for>"
}

Quality standards — follow these strictly:
- Every question must test a SPECIFIC objective from the lesson plan.
- Prefer application and reasoning over pure recall.
- When generating 3 or more questions, include at least one application/reasoning question.
- Math notation: use LaTeX. In JSON strings escape backslashes: \\\\( inline \\\\) and \\\\[ display \\\\].
- MCQ: provide exactly 4 options (A–D). Each distractor must target a LIKELY misconception — not random wrong answers.
- markscheme_steps marks must SUM to the total marks value. Split marks meaningfully by method/answer.
- worked_solution: show every step — not just the final answer.
- common_mistakes: 2 specific mistakes students commonly make on this exact question type.
- source_lesson_plan_excerpt: copy actual verbatim text from the approved lesson plan (not a paraphrase).
- options: always use an empty array [] for non-MCQ questions (never null or omit).

Return ONLY the JSON object. No markdown fences, no preamble, no postamble.
"""


# ── JSON schema for structured response (minimises broken output) ──────────────

_RICH_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions"],
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "slot", "title", "question_type", "difficulty", "marks",
                    "estimated_time_minutes", "question_text", "options",
                    "correct_answer", "worked_solution", "markscheme_steps",
                    "common_mistakes", "hints", "tags", "domain", "topic",
                    "subtopic", "exam_system", "subject", "level",
                    "source_lesson_plan_excerpt", "quality_notes",
                ],
                "properties": {
                    "slot":                      {"type": "integer"},
                    "title":                     {"type": "string"},
                    "question_type":             {"type": "string", "enum": ["multiple_choice", "short_answer", "long_answer", "fill_blank"]},
                    "difficulty":                {"type": "string", "enum": ["easy", "medium", "challenge"]},
                    "marks":                     {"type": "integer"},
                    "estimated_time_minutes":    {"type": "integer"},
                    "question_text":             {"type": "string"},
                    "options":                   {"type": "array", "items": {"type": "string"}},
                    "correct_answer":            {"type": "string"},
                    "worked_solution":           {"type": "string"},
                    "markscheme_steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["description", "marks"],
                            "properties": {
                                "description": {"type": "string"},
                                "marks":       {"type": "integer"},
                            },
                        },
                    },
                    "common_mistakes":            {"type": "array", "items": {"type": "string"}},
                    "hints":                      {"type": "array", "items": {"type": "string"}},
                    "tags":                       {"type": "array", "items": {"type": "string"}},
                    "domain":                     {"type": "string"},
                    "topic":                      {"type": "string"},
                    "subtopic":                   {"type": "string"},
                    "exam_system":                {"type": "string"},
                    "subject":                    {"type": "string"},
                    "level":                      {"type": "string"},
                    "source_lesson_plan_excerpt": {"type": "string"},
                    "quality_notes":              {"type": "string"},
                },
            },
        },
    },
}


# ── Rich question Pydantic models ──────────────────────────────────────────────

class MarkschemeStep(BaseModel):
    description: str
    marks: int = Field(default=1, ge=1, le=20)


class RichGeneratedQuestion(BaseModel):
    slot: int
    title: str = ""
    question_type: str
    difficulty: str
    marks: int = Field(default=2, ge=1, le=20)
    estimated_time_minutes: int = Field(default=3, ge=1, le=60)
    question_text: str
    options: list[str] = Field(default_factory=list)
    correct_answer: str = ""
    worked_solution: str = ""
    markscheme_steps: list[MarkschemeStep] = Field(default_factory=list)
    common_mistakes: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    domain: str = ""
    topic: str = ""
    subtopic: str = ""
    exam_system: str = ""
    subject: str = ""
    level: str = ""
    source_lesson_id: str = ""
    source_lesson_plan_excerpt: str = ""
    quality_notes: str = ""


class GenerateQuestionsRequest(BaseModel):
    lesson_id: str
    question_count: int = Field(default=5, ge=1, le=20)
    exclude_questions: list[str] | None = None
    generation_options: GenerationOptions | None = None

    @model_validator(mode="before")
    @classmethod
    def support_nested_question_count(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        opts = data.get("generation_options")
        if isinstance(opts, dict) and data.get("question_count") is None and opts.get("question_count") is not None:
            data = {**data, "question_count": opts.get("question_count")}
        return data

    @model_validator(mode="after")
    def validate_mix_totals(self) -> "GenerateQuestionsRequest":
        opts = self.generation_options
        if opts is None:
            return self
        count = self.question_count
        if opts.type_mix is not None:
            tm = opts.type_mix
            total = tm.multiple_choice + tm.short_answer + tm.long_answer + tm.fill_blank
            if total > count:
                raise ValueError(
                    f"type_mix total ({total}) exceeds question_count ({count})"
                )
        if opts.difficulty_mix is not None:
            dm = opts.difficulty_mix
            total = dm.easy + dm.medium + dm.challenge
            if total > count:
                raise ValueError(
                    f"difficulty_mix total ({total}) exceeds question_count ({count})"
                )
        tmt = opts.total_marks_target
        if tmt is not None:
            # Each question needs at least 1 mark
            if tmt < count:
                raise ValueError(
                    f"total_marks_target ({tmt}) is less than question_count ({count}); "
                    "each question needs at least 1 mark"
                )
            # Target must be reachable given marks_min
            if opts.marks_min is not None and tmt < count * opts.marks_min:
                raise ValueError(
                    f"total_marks_target ({tmt}) is impossible: "
                    f"{count} questions × marks_min {opts.marks_min} = {count * opts.marks_min} minimum"
                )
            # Target must be reachable given marks_max (or the absolute per-question cap of 20)
            effective_max = opts.marks_max if opts.marks_max is not None else 20
            if tmt > count * effective_max:
                raise ValueError(
                    f"total_marks_target ({tmt}) is impossible: "
                    f"{count} questions × marks_max {effective_max} = {count * effective_max} maximum"
                )
        return self


class GenerateQuestionsResponse(BaseModel):
    questions: list[RichGeneratedQuestion]
    lesson_title: str = ""
    source_meta: dict[str, Any] = Field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _repair_question(qr: Any, slot: int) -> RichGeneratedQuestion:
    """Parse and repair a raw question dict, filling gaps with safe defaults."""
    if not isinstance(qr, dict):
        return RichGeneratedQuestion(
            slot=slot, question_type="short_answer", difficulty="medium",
            question_text=f"Question {slot}", marks=2,
        )

    _valid_types = {"multiple_choice", "short_answer", "long_answer", "fill_blank"}
    _valid_diffs = {"easy", "medium", "challenge"}

    q_type = qr.get("question_type", "short_answer")
    if q_type not in _valid_types:
        q_type = "short_answer"

    difficulty = qr.get("difficulty", "medium")
    if difficulty not in _valid_diffs:
        difficulty = "medium"

    try:
        marks = max(1, min(20, int(qr.get("marks", 2))))
    except (TypeError, ValueError):
        marks = 2

    try:
        est_time = max(1, min(60, int(qr.get("estimated_time_minutes", 3))))
    except (TypeError, ValueError):
        est_time = 3

    # options: empty list for non-MCQ, coerce to list for MCQ
    raw_opts = qr.get("options")
    if q_type == "multiple_choice":
        options = [str(o) for o in raw_opts] if isinstance(raw_opts, list) else []
    else:
        options = []

    # markscheme_steps: parse and repair total-marks consistency
    raw_steps = qr.get("markscheme_steps") or []
    steps: list[MarkschemeStep] = []
    for step in raw_steps:
        if isinstance(step, dict):
            try:
                s_marks = max(1, int(step.get("marks", 1)))
            except (TypeError, ValueError):
                s_marks = 1
            steps.append(MarkschemeStep(
                description=str(step.get("description") or "").strip() or "Mark awarded",
                marks=s_marks,
            ))

    if steps:
        steps_total = sum(s.marks for s in steps)
        if steps_total < marks:
            steps.append(MarkschemeStep(
                description="Remaining marks for correct method/answer",
                marks=marks - steps_total,
            ))
        elif steps_total > marks:
            # Trim from the last step
            excess = steps_total - marks
            for i in range(len(steps) - 1, -1, -1):
                if steps[i].marks > excess:
                    steps[i] = MarkschemeStep(description=steps[i].description, marks=steps[i].marks - excess)
                    break
                elif steps[i].marks <= excess:
                    excess -= steps[i].marks
                    steps.pop(i)
                    if excess == 0:
                        break

    def _str_list(val: Any) -> list[str]:
        if isinstance(val, list):
            return [str(s).strip() for s in val if s]
        return []

    return RichGeneratedQuestion(
        slot=int(qr.get("slot", slot)),
        title=str(qr.get("title") or "").strip() or f"Question {slot}",
        question_type=q_type,
        difficulty=difficulty,
        marks=marks,
        estimated_time_minutes=est_time,
        question_text=str(qr.get("question_text") or "").strip() or f"Question {slot}",
        options=options,
        correct_answer=str(qr.get("correct_answer") or "").strip() or "See worked solution",
        worked_solution=str(qr.get("worked_solution") or "").strip(),
        markscheme_steps=steps,
        common_mistakes=_str_list(qr.get("common_mistakes")),
        hints=_str_list(qr.get("hints")),
        tags=_str_list(qr.get("tags")),
        domain=str(qr.get("domain") or "").strip(),
        topic=str(qr.get("topic") or "").strip(),
        subtopic=str(qr.get("subtopic") or "").strip(),
        exam_system=str(qr.get("exam_system") or "").strip(),
        subject=str(qr.get("subject") or "").strip(),
        level=str(qr.get("level") or "").strip(),
        source_lesson_plan_excerpt=str(qr.get("source_lesson_plan_excerpt") or "").strip(),
        quality_notes=str(qr.get("quality_notes") or "").strip(),
    )


async def _call_with_structured_questions(
    messages: list[dict[str, Any]],
    model: str,
) -> tuple[list[Any], Any]:
    """
    Call OpenRouter for question generation, cascading through JSON modes.
    Returns (questions_raw_list, result).
    """
    # Attempt 1: strict JSON schema — strongest output guarantee
    try:
        result = await call_openrouter(
            model=model,
            messages=messages,
            temperature=GENERATE_QUESTIONS_TEMPERATURE,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "practice_questions",
                    "strict": True,
                    "schema": _RICH_QUESTION_SCHEMA,
                },
            },
        )
        parsed = json.loads(strip_json_fences(result.content or ""))
        if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
            return parsed["questions"], result
        if isinstance(parsed, list):
            return parsed, result
    except Exception as exc:
        logger.warning("[generate-questions] strict schema attempt failed: %s", exc)

    # Attempt 2: json_object mode
    try:
        result = await call_openrouter(
            model=model,
            messages=messages,
            temperature=GENERATE_QUESTIONS_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(strip_json_fences(result.content or ""))
        if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
            return parsed["questions"], result
        if isinstance(parsed, list):
            return parsed, result
    except Exception as exc:
        logger.warning("[generate-questions] json_object attempt failed: %s", exc)

    # Attempt 3: plain call + parse
    result = await call_openrouter(
        model=model,
        messages=messages,
        temperature=GENERATE_QUESTIONS_TEMPERATURE,
    )
    raw = strip_json_fences(result.content or "")
    parsed = json.loads(raw)  # propagate JSONDecodeError as HTTP 502
    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
        return parsed["questions"], result
    if isinstance(parsed, list):
        return parsed, result
    raise ValueError(f"Unexpected response shape: {type(parsed).__name__}")


@router.post(
    "/generate-questions",
    response_model=GenerateQuestionsResponse,
    summary="Generate practice questions from the approved lesson plan",
)
async def generate_practice_questions(
    body: GenerateQuestionsRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher")),
    ],
) -> GenerateQuestionsResponse:
    request_id = str(uuid.uuid4())[:8]
    svc = get_service_client()

    logger.info(
        "[generate-questions][%s] user=%s lesson=%s count=%d",
        request_id, user.id, body.lesson_id, body.question_count,
    )

    lesson = await _verify_lesson_access(lesson_id=body.lesson_id, user=user, svc=svc, owner_only=True)
    lesson_plan = (lesson.get("lesson_plan") or "").strip()
    if not lesson_plan:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Lesson plan is empty — approve a lesson plan first",
        )
    if lesson.get("lesson_plan_status") != "approved":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Approve the lesson plan before generating practice questions",
        )

    db_ctx = await _fetch_lesson_context(lesson=lesson, svc=svc)
    context_lines = [f"Lesson title: {lesson.get('title') or 'Untitled'}"]
    if db_ctx.get("className"):
        context_lines.append(f"Class: {db_ctx['className']}")
    students = db_ctx.get("students") or []
    if students:
        context_lines.append(f"Students: {', '.join(s['name'] for s in students[:5])}")

    system = (
        _GENERATE_QUESTIONS_SYSTEM
        + f"\n\n## Lesson context\n" + "\n".join(context_lines)
        + f"\n\n## Approved lesson plan\n{lesson_plan[:6000]}"
    )

    if body.generation_options:
        opts_block = _build_options_block(body.generation_options, body.question_count)
        if opts_block:
            system += f"\n\n## Generation requirements (follow precisely)\n{opts_block}"

    user_msg = (
        f"Generate exactly {body.question_count} practice question"
        f"{'s' if body.question_count != 1 else ''} from the lesson plan above."
    )
    if body.exclude_questions:
        excluded = "\n".join(f"- {q}" for q in body.exclude_questions[:20])
        user_msg += f"\n\nDo NOT reuse or closely paraphrase any of these existing questions:\n{excluded}"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    caller_id = user.id or "internal"
    route = "POST /v1/online-lesson/generate-questions"
    ai_usage: AIUsage | None = None

    try:
        questions_raw, result = await _call_with_structured_questions(
            messages=messages,
            model=settings.online_lesson_questions_model,
        )
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
    except json.JSONDecodeError as exc:
        logger.error("[generate-questions][%s] JSON parse failed: %s", request_id, exc)
        await record_ai_usage(
            svc, user_id=caller_id, source="online_lesson_generate_questions",
            entity_type="lesson", entity_id=body.lesson_id,
            route=route, status="error", error_message=f"JSON: {exc}",
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned unexpected output — please try again",
        ) from exc
    except Exception as exc:
        logger.error("[generate-questions][%s] OpenRouter error: %s", request_id, exc)
        await record_ai_usage(
            svc, user_id=caller_id, source="online_lesson_generate_questions",
            entity_type="lesson", entity_id=body.lesson_id,
            route=route, status="error", error_message=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service unavailable — please try again",
        ) from exc

    await record_ai_usage(
        svc, user_id=caller_id, source="online_lesson_generate_questions",
        entity_type="lesson", entity_id=body.lesson_id,
        route=route, usage=ai_usage, status="success",
    )

    # Repair each question (fills missing fields, validates enums, fixes markscheme totals)
    questions: list[RichGeneratedQuestion] = []
    for i, qr in enumerate(questions_raw[: body.question_count]):
        try:
            q = _repair_question(qr, slot=i + 1)
            q.source_lesson_id = body.lesson_id
            questions.append(q)
        except Exception as exc:
            logger.warning("[generate-questions][%s] question %d repair failed: %s", request_id, i, exc)
            questions.append(RichGeneratedQuestion(
                slot=i + 1,
                question_type="short_answer",
                difficulty="medium",
                question_text=f"Question {i + 1}",
                marks=2,
                source_lesson_id=body.lesson_id,
            ))

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No questions were generated — please try again",
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    plan_meta = lesson.get("lesson_plan_source_meta") if isinstance(lesson.get("lesson_plan_source_meta"), dict) else {}
    source_meta = {
        **_standard_artifact_meta(
            lesson=lesson,
            teacher_id=user.id,
            model=ai_usage.model,
            prompt_version="online_lesson_practice_questions_v1",
            generated_at=generated_at,
            approved_at=lesson.get("lesson_plan_approved_at"),
            source_plan_hash=str(plan_meta.get("sourcePlanHash") or _hash_text(lesson_plan)),
            chat_history_turns=(
                int(plan_meta["chatHistoryTurns"])
                if isinstance(plan_meta.get("chatHistoryTurns"), int)
                else None
            ),
        ),
        "prompt_tokens": ai_usage.prompt_tokens,
        "completion_tokens": ai_usage.completion_tokens,
        "generation_options": body.generation_options.model_dump() if body.generation_options else {
            "question_count": body.question_count,
        },
    }

    logger.info(
        "[generate-questions][%s] done questions=%d tokens=%s/%s",
        request_id, len(questions), ai_usage.prompt_tokens, ai_usage.completion_tokens,
    )

    return GenerateQuestionsResponse(
        questions=questions,
        lesson_title=lesson.get("title") or "",
        source_meta=source_meta,
    )
