"""
Online lesson AI endpoints:

  POST /v1/online-lesson/chat           — lesson planning conversation
  POST /v1/online-lesson/generate-plan  — generate a full lesson plan from chat context
  POST /v1/online-lesson/revise-plan    — revise an existing plan based on teacher feedback

Auth: lesson owners can mutate planning resources; co-teachers may read elsewhere.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import call_openrouter, strip_json_fences
from gradenza_api.services.supabase_client import get_service_client, run_sync
from gradenza_api.services.usage import AIUsage, record_ai_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/online-lesson", tags=["online-lessons"])

CHAT_MODEL = "google/gemini-2.5-flash"
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

Your job is to gather information about the planned lesson through natural conversational exchange,
then help the teacher feel ready to generate a high-quality lesson plan.

You MUST respond with a JSON object in this exact format:
{
  "reply": "<conversational message to the teacher — 2–4 sentences max>",
  "readyToGenerateLessonPlan": <true or false>,
  "missingContext": ["<thing 1>", "<thing 2>"],
  "detectedTopic": "<main topic or empty string if unknown>",
  "detectedObjectives": ["<objective 1>", "<objective 2>"],
  "suggestedQuestionCount": <integer, or 0 if topic not yet clear>
}

Field rules:
- reply: warm, encouraging, focused. Ask one or two concrete questions at a time.
- readyToGenerateLessonPlan: set true only when ALL three are clear: topic, student level, and at least one learning objective.
- missingContext: list up to 4 things that are still missing (empty when ready).
- detectedTopic: best guess at the topic/concept to teach, or "" if still unknown.
- detectedObjectives: specific learning outcomes detected so far (may be empty early on).
- suggestedQuestionCount: sensible number of practice questions for the scope (0 if topic unclear).

Information to gather through conversation:
1. Topic or concept to teach
2. Student level / year / exam system (e.g. IB HL, SAT, GCSE)
3. Learning objective(s)
4. Weak areas or prerequisite gaps
5. Lesson duration
6. Preferred teaching style (worked examples, discovery, etc.)

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
            .select("id, teacher_id, kind, class_id, title, lesson_plan, lesson_plan_status, general_notes")
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
    try:
        result = await call_openrouter(
            model=CHAT_MODEL,
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
            model=CHAT_MODEL,
            messages=messages,
            temperature=CHAT_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        return json.loads(strip_json_fences(result.content)), result
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("[online-lesson-chat] json_object mode failed: %s", exc)

    # Attempt 3: plain call
    result = await call_openrouter(
        model=CHAT_MODEL,
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

GENERATE_PLAN_MODEL = "google/gemini-2.5-flash"
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
            model=GENERATE_PLAN_MODEL,
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

    source_meta = {
        "model": result.usage.model,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "history_turns": len(history),
        "lesson_title": lesson.get("title"),
        "students": [s["name"] for s in (db_ctx.get("students") or [])],
        "class_name": db_ctx.get("className"),
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

REVISE_PLAN_MODEL = "google/gemini-2.5-flash"
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
            model=REVISE_PLAN_MODEL,
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

GENERATE_QUESTIONS_MODEL = "google/gemini-2.5-flash"
GENERATE_QUESTIONS_TEMPERATURE = 0.5

_GENERATE_QUESTIONS_SYSTEM = """\
You are an expert quiz creator. Generate practice questions based on a teacher's approved lesson plan.

Questions must:
- Cover the topics, objectives, definitions, and worked examples from the lesson plan
- Be exam-appropriate and clearly phrased
- Mix types: prefer short_answer and multiple_choice; use long_answer for deeper tasks
- Use medium difficulty unless the plan implies a specific level
- Use LaTeX for math: \\( inline \\) and \\[ display \\]. In JSON strings escape: \\\\( ... \\\\)

Return a JSON object with key "questions" containing an array.
Each question object:
{
  "slot": <integer starting at 1>,
  "question_type": "multiple_choice" | "short_answer" | "long_answer" | "fill_blank",
  "difficulty": "easy" | "medium" | "challenge",
  "question_text": "<full question text>",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."] for multiple_choice, null otherwise,
  "correct_answer": "<letter key for MC, or full answer text>",
  "explanation": "<brief explanation or marking guidance>",
  "marks": <integer 1-5>
}

Return ONLY the JSON object. No markdown fences, no text outside the JSON.
"""


class GenerateQuestionsRequest(BaseModel):
    lesson_id: str
    question_count: int = Field(default=5, ge=1, le=20)
    exclude_questions: list[str] | None = None


class GeneratedQuestion(BaseModel):
    slot: int
    question_type: str
    difficulty: str
    question_text: str
    options: list[str] | None = None
    correct_answer: str
    explanation: str = ""
    marks: int = 2


class GenerateQuestionsResponse(BaseModel):
    questions: list[GeneratedQuestion]
    lesson_title: str = ""
    source_meta: dict[str, Any] = Field(default_factory=dict)


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
        result = await call_openrouter(
            model=GENERATE_QUESTIONS_MODEL,
            messages=messages,
            temperature=GENERATE_QUESTIONS_TEMPERATURE,
            response_format={"type": "json_object"},
        )
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
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

    raw = strip_json_fences(result.content or "")
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("[generate-questions][%s] JSON parse failed: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned unexpected output — please try again",
        ) from exc

    # Model may return {"questions": [...]} or directly [...]
    if isinstance(parsed, list):
        questions_raw: list[Any] = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
        questions_raw = parsed["questions"]
    else:
        logger.error("[generate-questions][%s] unexpected shape: %s", request_id, type(parsed).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI returned unexpected output format — please try again",
        )

    _valid_types = {"multiple_choice", "short_answer", "long_answer", "fill_blank"}
    _valid_diffs = {"easy", "medium", "challenge"}

    questions: list[GeneratedQuestion] = []
    for i, qr in enumerate(questions_raw[: body.question_count]):
        try:
            q = GeneratedQuestion(
                slot=int(qr.get("slot", i + 1)),
                question_type=qr.get("question_type", "short_answer")
                if qr.get("question_type") in _valid_types else "short_answer",
                difficulty=qr.get("difficulty", "medium")
                if qr.get("difficulty") in _valid_diffs else "medium",
                question_text=str(qr.get("question_text") or "").strip() or f"Question {i + 1}",
                options=qr.get("options") if isinstance(qr.get("options"), list) else None,
                correct_answer=str(qr.get("correct_answer") or "").strip() or "See explanation",
                explanation=str(qr.get("explanation") or "").strip(),
                marks=max(1, min(5, int(qr.get("marks", 2)))),
            )
            questions.append(q)
        except Exception as exc:
            logger.warning("[generate-questions][%s] question %d parse error: %s", request_id, i, exc)
            questions.append(GeneratedQuestion(
                slot=i + 1,
                question_type="short_answer",
                difficulty="medium",
                question_text=f"Question {i + 1}",
                correct_answer="See explanation",
                explanation="",
                marks=2,
            ))

    if not questions:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No questions were generated — please try again",
        )

    source_meta = {
        "model": ai_usage.model,
        "prompt_tokens": ai_usage.prompt_tokens,
        "completion_tokens": ai_usage.completion_tokens,
        "lesson_id": body.lesson_id,
        "lesson_title": lesson.get("title") or "",
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
