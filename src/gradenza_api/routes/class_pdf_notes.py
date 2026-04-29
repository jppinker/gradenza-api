"""
POST /v1/materials/class-pdf-notes/chat

Chat-driven Class PDF Notes generator.

Actions:
  chat              — conversational guidance, returns text
  generate_pdf_notes — return a complete structured notes document
  regenerate_section — return full notes with one section updated
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import OpenRouterResult, call_openrouter, strip_json_fences
from gradenza_api.services.supabase_client import get_service_client
from gradenza_api.services.usage import AIUsage, record_ai_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/materials/class-pdf-notes", tags=["class-pdf-notes"])

PDF_NOTES_MODEL = "google/gemini-2.5-flash"
CHAT_TEMPERATURE = 0.7
GENERATE_TEMPERATURE = 0.5
MAX_MESSAGES = 20

# ── Pydantic models ───────────────────────────────────────────────────────────


class ContentCreatorMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ClassPdfNotesSection(BaseModel):
    title: str
    contentMarkdown: str
    workedExampleMarkdown: str | None = None
    graphExpressions: list[str] = Field(default_factory=list)
    graphCaption: str | None = None


class ClassPdfNotes(BaseModel):
    schemaVersion: int = 1
    title: str
    subtitle: str
    overviewMarkdown: str
    learningObjectives: list[str]
    sections: list[ClassPdfNotesSection]
    practiceChecklist: list[str]
    studentSummaryMarkdown: str
    teacherNote: str


class ClassPdfNotesChatRequest(BaseModel):
    action: Literal["chat", "generate_pdf_notes", "regenerate_section"]
    messages: list[ContentCreatorMessage] = Field(default_factory=list)
    classContext: dict[str, Any] | None = None
    existingDraft: dict[str, Any] | None = None
    sectionIndex: int | None = None
    sectionFeedback: str | None = None


class ClassPdfNotesChatResponse(BaseModel):
    type: Literal["chat", "notes"]
    text: str | None = None
    notes: ClassPdfNotes | None = None
    model: str | None = None


# ── JSON schema for structured output ────────────────────────────────────────

_PDF_NOTES_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schemaVersion": {"type": "integer"},
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "overviewMarkdown": {"type": "string"},
        "learningObjectives": {"type": "array", "items": {"type": "string"}},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "contentMarkdown": {"type": "string"},
                    "workedExampleMarkdown": {"type": "string"},
                    "graphExpressions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3,
                    },
                    "graphCaption": {"type": "string"},
                },
                "required": ["title", "contentMarkdown", "graphExpressions"],
                "additionalProperties": False,
            },
        },
        "practiceChecklist": {"type": "array", "items": {"type": "string"}},
        "studentSummaryMarkdown": {"type": "string"},
        "teacherNote": {"type": "string"},
    },
    "required": [
        "schemaVersion",
        "title",
        "subtitle",
        "overviewMarkdown",
        "learningObjectives",
        "sections",
        "practiceChecklist",
        "studentSummaryMarkdown",
        "teacherNote",
    ],
    "additionalProperties": False,
}

# ── System prompts ────────────────────────────────────────────────────────────

_CHAT_SYSTEM = """\
You are an expert educational content designer helping a teacher create class PDF notes.
Your role is to ask focused clarifying questions and guide the teacher toward generating
a complete, printable handout. Keep responses concise (2–4 sentences). After 2–3 messages
with enough context (topic, exam system, student level), invite the teacher to generate the notes.
Do not generate a full document in this mode — just conversational guidance.
"""

_GENERATE_SYSTEM = """\
You are an expert educational content designer. Create a complete, student-facing class PDF \
handout as a JSON object matching the schema exactly.

Rules:
- Write for students, not a generic audience. Assume they are studying for the specified exam.
- Use Markdown for all content fields. Use KaTeX-compatible math: \\( inline \\) and \\[ block \\].
- graphExpressions: plain functions of x only (e.g. "x^2 - 3*x + 2"). No "y=", no LaTeX, no \
inequalities, no piecewise. Up to 3 per section. Empty array if not needed.
- teacherNote: teacher-only pedagogy advice, common misconceptions, marking tips. Not for students.
- learningObjectives: 3–5 bullet-point goals students should achieve.
- practiceChecklist: 4–6 self-check items students can tick after studying.
- subtitle: one-line topic summary.
- Do not reproduce copyrighted material verbatim. Generate original explanations.
- schemaVersion must be 1.
- Return only the JSON object. No markdown fences around it.
"""

_REGENERATE_SECTION_SYSTEM = """\
You are updating one section of an existing class PDF handout. Return the COMPLETE notes JSON \
with every field preserved from the original — only modify the section at the requested index \
based on the teacher's feedback. Keep the same number of sections, title, subtitle, objectives, \
checklist, summary, and teacher note unless the feedback explicitly asks to change them.
Return only the JSON object. No markdown fences.
"""

# ── JSON generation helper ────────────────────────────────────────────────────


async def _call_with_json_output(
    messages: list[dict[str, Any]],
    temperature: float,
) -> tuple[dict[str, Any], OpenRouterResult]:
    """
    Try to get valid JSON notes from the model with three fallback levels:
    1. Strict JSON schema output
    2. JSON object mode
    3. Plain call + repair prompt
    """
    last_result: OpenRouterResult | None = None
    last_raw = ""

    # Attempt 1: strict JSON schema
    try:
        result = await call_openrouter(
            model=PDF_NOTES_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "class_pdf_notes",
                    "strict": True,
                    "schema": _PDF_NOTES_JSON_SCHEMA,
                },
            },
        )
        last_result = result
        last_raw = strip_json_fences(result.content)
        return json.loads(last_raw), result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("class_pdf_notes strict schema failed: %s", e)

    # Attempt 2: json_object mode
    try:
        result = await call_openrouter(
            model=PDF_NOTES_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        last_result = result
        last_raw = strip_json_fences(result.content)
        return json.loads(last_raw), result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("class_pdf_notes json_object mode failed: %s", e)

    # Attempt 3: plain call + repair prompt
    repair_messages = list(messages) + [
        {"role": "assistant", "content": last_raw or "(no output)"},
        {
            "role": "user",
            "content": (
                "The previous response was not valid JSON. "
                "Return only the JSON object, no markdown fences or other text."
            ),
        },
    ]
    result = await call_openrouter(
        model=PDF_NOTES_MODEL,
        messages=repair_messages,
        temperature=0.0,
    )
    raw = strip_json_fences(result.content)
    return json.loads(raw), result  # allow JSONDecodeError to propagate as 502


# ── Class context builder ─────────────────────────────────────────────────────


def _format_class_context(ctx: dict[str, Any] | None) -> str:
    if not ctx:
        return ""
    parts = []
    if ctx.get("className"):
        parts.append(f"Class: {ctx['className']}")
    if ctx.get("examSystem"):
        parts.append(f"Exam system: {ctx['examSystem']}")
    if ctx.get("examSubject"):
        parts.append(f"Subject: {ctx['examSubject']}")
    if ctx.get("topicHints"):
        parts.append(f"Topic hints: {ctx['topicHints']}")
    return "\n".join(parts)


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post(
    "/chat",
    response_model=ClassPdfNotesChatResponse,
    summary="Chat-driven Class PDF notes generation",
)
async def class_pdf_notes_chat(
    body: ClassPdfNotesChatRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher")),
    ],
) -> ClassPdfNotesChatResponse:
    request_id = str(uuid.uuid4())[:8]
    logger.info(
        "[class-pdf-notes-chat][%s] user=%s action=%s",
        request_id,
        user.id,
        body.action,
    )

    svc = get_service_client()
    ai_usage: AIUsage | None = None
    route = "POST /v1/materials/class-pdf-notes/chat"

    messages = body.messages[-MAX_MESSAGES:]
    class_context_text = _format_class_context(body.classContext)

    # ── chat ─────────────────────────────────────────────────────

    if body.action == "chat":
        system = _CHAT_SYSTEM
        if class_context_text:
            system += f"\n\nClass context:\n{class_context_text}"

        or_messages = [{"role": "system", "content": system}]
        for m in messages:
            or_messages.append({"role": m.role, "content": m.content})

        try:
            result = await call_openrouter(
                model=PDF_NOTES_MODEL,
                messages=or_messages,
                temperature=CHAT_TEMPERATURE,
            )
            ai_usage = AIUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                model=result.usage.model,
                request_id=result.usage.request_id,
            )
        except Exception as exc:
            logger.error("[class-pdf-notes-chat][%s] chat error: %s", request_id, exc)
            await record_ai_usage(
                svc, user_id=user.id or "", source="class_pdf_notes",
                route=route, status="error", error_message=str(exc),
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Generation service error") from exc

        await record_ai_usage(
            svc, user_id=user.id or "", source="class_pdf_notes",
            entity_type="chat", route=route, usage=ai_usage, status="success",
        )
        return ClassPdfNotesChatResponse(type="chat", text=result.content, model=result.usage.model)

    # ── generate_pdf_notes ───────────────────────────────────────

    if body.action == "generate_pdf_notes":
        system = _GENERATE_SYSTEM
        if class_context_text:
            system += f"\n\nClass context:\n{class_context_text}"

        or_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            or_messages.append({"role": m.role, "content": m.content})

        try:
            parsed, result = await _call_with_json_output(or_messages, GENERATE_TEMPERATURE)
            ai_usage = AIUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                model=result.usage.model,
                request_id=result.usage.request_id,
            )
        except json.JSONDecodeError as exc:
            logger.error("[class-pdf-notes-chat][%s] JSON parse failed: %s", request_id, exc)
            await record_ai_usage(
                svc, user_id=user.id or "", source="class_pdf_notes",
                route=route, status="error", error_message=f"JSON parse: {exc}",
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Generation returned invalid JSON") from exc
        except Exception as exc:
            logger.error("[class-pdf-notes-chat][%s] generate error: %s", request_id, exc)
            await record_ai_usage(
                svc, user_id=user.id or "", source="class_pdf_notes",
                route=route, status="error", error_message=str(exc),
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Generation service error") from exc

        await record_ai_usage(
            svc, user_id=user.id or "", source="class_pdf_notes",
            entity_type="pdf_notes", route=route, usage=ai_usage, status="success",
        )

        parsed.setdefault("schemaVersion", 1)
        notes = ClassPdfNotes(**{k: parsed[k] for k in ClassPdfNotes.model_fields if k in parsed})
        return ClassPdfNotesChatResponse(type="notes", notes=notes, model=result.usage.model)

    # ── regenerate_section ───────────────────────────────────────

    if body.action == "regenerate_section":
        if body.existingDraft is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="existingDraft required for regenerate_section")
        if body.sectionIndex is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="sectionIndex required for regenerate_section")

        existing_sections = body.existingDraft.get("sections") or []
        original_count = len(existing_sections)
        idx = body.sectionIndex

        if idx < 0 or idx >= original_count:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="sectionIndex out of range")

        system = _REGENERATE_SECTION_SYSTEM
        if class_context_text:
            system += f"\n\nClass context:\n{class_context_text}"

        draft_json = json.dumps(body.existingDraft, ensure_ascii=False)
        feedback_text = body.sectionFeedback or "Please improve this section."
        user_instruction = (
            f"Regenerate only section at index {idx} (0-based) of the following notes.\n"
            f"Teacher feedback: {feedback_text}\n\n"
            f"Current notes:\n{draft_json}"
        )

        or_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_instruction},
        ]

        try:
            parsed, result = await _call_with_json_output(or_messages, GENERATE_TEMPERATURE)
            ai_usage = AIUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                model=result.usage.model,
                request_id=result.usage.request_id,
            )
        except json.JSONDecodeError as exc:
            logger.error("[class-pdf-notes-chat][%s] regen JSON parse failed: %s", request_id, exc)
            await record_ai_usage(
                svc, user_id=user.id or "", source="class_pdf_notes",
                route=route, status="error", error_message=f"regen JSON parse: {exc}",
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Generation returned invalid JSON") from exc
        except Exception as exc:
            logger.error("[class-pdf-notes-chat][%s] regen error: %s", request_id, exc)
            await record_ai_usage(
                svc, user_id=user.id or "", source="class_pdf_notes",
                route=route, status="error", error_message=str(exc),
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Generation service error") from exc

        await record_ai_usage(
            svc, user_id=user.id or "", source="class_pdf_notes",
            entity_type="pdf_notes_regen", route=route, usage=ai_usage, status="success",
        )

        returned_sections = parsed.get("sections") or []

        # Merge: if section count changed, splice only the regenerated section into the original
        if len(returned_sections) != original_count:
            logger.warning(
                "[class-pdf-notes-chat][%s] section count mismatch original=%d returned=%d — merging",
                request_id, original_count, len(returned_sections),
            )
            merged_draft = dict(body.existingDraft)
            merged_sections = list(existing_sections)
            replacement = returned_sections[0] if returned_sections else existing_sections[idx]
            merged_sections[idx] = replacement
            merged_draft["sections"] = merged_sections
            parsed = merged_draft
        else:
            # Sections match — but still splice only the changed section to prevent accidental loss
            merged_draft = dict(body.existingDraft)
            merged_sections = list(existing_sections)
            merged_sections[idx] = returned_sections[idx]
            merged_draft["sections"] = merged_sections
            # Preserve all other top-level fields from the returned notes (title etc may have improved)
            for k, v in parsed.items():
                if k != "sections":
                    merged_draft[k] = v
            parsed = merged_draft

        parsed.setdefault("schemaVersion", 1)
        notes = ClassPdfNotes(**{k: parsed[k] for k in ClassPdfNotes.model_fields if k in parsed})
        return ClassPdfNotesChatResponse(type="notes", notes=notes, model=result.usage.model)

    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown action")
