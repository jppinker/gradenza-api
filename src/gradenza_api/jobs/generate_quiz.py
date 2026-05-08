"""
ARQ job: generate_quiz(ctx, session_id)

Generates quiz questions for a quiz_builder_sessions row:
1. Fetch session, source material, blueprint.
2. Call LLM to generate questions matching the blueprint spec.
3. Save questions to quiz_questions.
4. Update session status to 'review'.
Sets status='error' on any failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from gradenza_api.services.llm_structured import (
    StructuredJSONError,
    call_openrouter_structured_json,
)
from gradenza_api.services.openrouter import call_openrouter, strip_json_fences
from gradenza_api.services.quiz_schemas import QuizQuestionSchema
from gradenza_api.services.supabase_client import get_service_client
from gradenza_api.services.usage import AIUsage, record_ai_usage

logger = logging.getLogger(__name__)

_QUIZ_MODEL = "google/gemini-2.5-flash"

_DEFAULT_DIFFICULTY_DEFS: dict[str, str] = {
    "easy": "Straightforward recall or basic comprehension; suitable for checking core understanding.",
    "medium": "Requires explanation, comparison, or applying the content in a familiar context.",
    "challenge": "Requires deeper reasoning, synthesis, evaluation, or applying the content in a less familiar context.",
}

_GENERATE_PROMPT = """\
You are an expert quiz creator. Generate high-quality quiz questions.

Session context:
- Subject: {subject}
- Grade level: {grade_level}
- Mode: {mode}
- Preset: {preset}

Source material:
{source_text}

Difficulty definitions — apply these when generating questions at each level:
- Easy: {easy_def}
- Medium: {medium_def}
- Challenge: {challenge_def}

LaTeX / KaTeX math & scientific notation (store LaTeX source text, not HTML) — REQUIRED whenever the source or question contains formula-like notation (even if Subject is "General").
Apply to ALL fields: question_text/options/correct_answer/acceptable_answers/explanation/rubric.
At minimum, question_text/options/correct_answer/explanation/rubric must follow these rules.
- Delimiters in the stored text: \\( ... \\) for inline, \\[ ... \\] for display.
- JSON escaping: you are returning JSON, so backslashes must be escaped. In your JSON output strings write \\\\( ... \\\\) and \\\\[ ... \\\\].
  Example JSON string value: "\\\\( \\\\frac{{d}}{{dx}}\\\\sin x = \\\\cos x \\\\)"
- Convert EVERY formula-like snippet (derivative, integral, limit, equation, symbol expression, unit expression, exponent/subscript, trig identity, scientific notation) into KaTeX-compatible LaTeX and wrap it in delimiters. Do NOT leave ASCII math.
- Forbidden ASCII math examples (do not output these): d/dx sin(x) = cos(x); sin^2(x) + cos^2(x) = 1
  Required LaTeX (stored text): \\( \\frac{{d}}{{dx}}\\sin x = \\cos x \\); \\( \\sin^2 x + \\cos^2 x = 1 \\)
- Use LaTeX syntax (KaTeX-friendly): fractions \\frac{{a}}{{b}}, roots \\sqrt{{x}}, subscripts x_{{n}}, superscripts x^{{n}}, Greek \\alpha \\beta \\gamma, multiplication \\times or \\cdot, units \\text{{m/s}}.
- Never output HTML tags (e.g., <span>...</span>) or KaTeX/MathJax-generated HTML; only LaTeX source inside delimiters.

Blueprint — generate EXACTLY these questions:
{blueprint}

Return a JSON array where each element follows this schema:
{{
  "slot": <integer matching blueprint slot>,
  "question_type": "multiple_choice" | "fill_blank" | "short_answer" | "long_answer",
  "difficulty": "easy" | "medium" | "challenge",
  "bloom_level": "remember" | "understand" | "apply" | "analyse" | "evaluate" | "create",
  "question_text": "<full question text>",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."] for MC, null otherwise,
  "correct_answer": "<answer letter A/B/C/D for MC, word/phrase for fill_blank, model answer otherwise>",
  "acceptable_answers": ["<alt answer>"] or null,
  "explanation": "<brief explanation>",
  "marks": <integer matching blueprint>,
  "rubric": "<rubric for long_answer>" or null,
  "answer_space": "multiple_choice" | "single_line" | "short_paragraph" | "long_paragraph",
  "source_reference": null,
  "image_slot": null
}}

answer_space mapping: multiple_choice→"multiple_choice", fill_blank→"single_line", \
short_answer→"short_paragraph", long_answer→"long_paragraph"

Return ONLY the raw JSON array. No markdown fences, no explanation."""


async def generate_quiz(ctx: dict, *, session_id: str) -> None:
    job_id = str(ctx.get("job_id") or "")
    logger.info("[generate_quiz] job started session=%s job_id=%s", session_id, job_id)
    svc = get_service_client()

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _touch_session(updates: dict[str, Any]) -> None:
        svc.table("quiz_builder_sessions").update({**updates, "updated_at": _now_iso()}).eq("id", session_id).execute()

    def _still_current_job() -> bool:
        """Check whether this worker's job_id still matches the session's generation_job_id.

        This MUST be checked before any terminal/destructive writes (failed/completed, or
        deleting/inserting quiz_questions) to prevent superseded workers from overwriting retries.
        """
        if not job_id:
            return True
        current = (
            svc.table("quiz_builder_sessions")
            .select("generation_job_id")
            .eq("id", session_id)
            .maybe_single()
            .execute()
        ).data
        current_job_id = str((current or {}).get("generation_job_id") or "")
        return current_job_id == job_id

    def _fetch() -> tuple[dict | None, dict | None, dict | None]:
        sess = (
            svc.table("quiz_builder_sessions")
            .select("*")
            .eq("id", session_id)
            .maybe_single()
            .execute()
        ).data

        src = (
            svc.table("quiz_source_materials")
            .select("extracted_text")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ).data
        src = src[0] if src else None

        bp = (
            svc.table("quiz_blueprints")
            .select("blueprint_json, difficulty_definitions")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ).data
        bp = bp[0] if bp else None

        return sess, src, bp

    session, source, blueprint = await asyncio.to_thread(_fetch)

    if not session:
        logger.error("[generate_quiz] session not found session=%s job_id=%s", session_id, job_id)
        return

    current_job_id = str(session.get("generation_job_id") or "")
    if current_job_id and job_id and current_job_id != job_id:
        logger.info(
            "[generate_quiz] superseded job ignored session=%s job_id=%s current_job_id=%s",
            session_id,
            job_id,
            current_job_id,
        )
        return

    await asyncio.to_thread(_touch_session, {
        "status": "generating",
        "error_message": None,
        "generation_status": "generating",
        "generation_stage": "preparing_source",
        "generation_failed_at": None,
        "generation_completed_at": None,
        "generation_error_code": None,
        "generation_error_message": None,
    })

    source_text: str = (source or {}).get("extracted_text") or "(no source text provided)"
    blueprint_items: list[Any] = (blueprint or {}).get("blueprint_json") or []
    raw_defs = (blueprint or {}).get("difficulty_definitions") or {}
    defs = {
        "easy": raw_defs.get("easy") or _DEFAULT_DIFFICULTY_DEFS["easy"],
        "medium": raw_defs.get("medium") or _DEFAULT_DIFFICULTY_DEFS["medium"],
        "challenge": raw_defs.get("challenge") or _DEFAULT_DIFFICULTY_DEFS["challenge"],
    }

    prompt = _GENERATE_PROMPT.format(
        subject=session.get("subject") or "General",
        grade_level=session.get("grade_level") or "Secondary",
        mode=session.get("mode", "from_prompt"),
        preset=session.get("preset", "standard_worksheet"),
        source_text=source_text[:3000],
        blueprint=json.dumps(blueprint_items, indent=2),
        easy_def=defs["easy"],
        medium_def=defs["medium"],
        challenge_def=defs["challenge"],
    )

    teacher_id: str = session.get("teacher_id") or ""

    try:
        expected_count = len(blueprint_items)

        def _summarize_validation_error(exc: ValidationError) -> str:
            try:
                first = (exc.errors() or [{}])[0]
                loc = ".".join(str(p) for p in first.get("loc", []) if p is not None) or "unknown"
                typ = first.get("type") or "validation_error"
                msg = first.get("msg") or "invalid value"
                return f"{loc}: {msg} ({typ})"
            except Exception:
                return "validation_error"

        def _validate_questions(obj: Any) -> list[dict[str, Any]]:
            if not isinstance(obj, list):
                raise ValueError(f"Expected JSON array of questions, got {type(obj).__name__}")
            if len(obj) != expected_count:
                raise ValueError(f"Expected {expected_count} questions, got {len(obj)}")
            out: list[dict[str, Any]] = []
            for i, q in enumerate(obj):
                try:
                    out.append(QuizQuestionSchema.model_validate(q).model_dump())
                except ValidationError as exc:
                    raise ValueError(f"Question {i} failed schema validation: {_summarize_validation_error(exc)}") from exc
            return out

        await asyncio.to_thread(_touch_session, {"generation_stage": "calling_llm"})

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "quiz_questions",
                "strict": True,
                "schema": {
                    "type": "array",
                    "minItems": expected_count,
                    "maxItems": expected_count,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "slot": {"type": "integer", "minimum": 1},
                            "question_type": {
                                "type": "string",
                                "enum": ["multiple_choice", "fill_blank", "short_answer", "long_answer"],
                            },
                            "difficulty": {"type": "string", "enum": ["easy", "medium", "challenge"]},
                            "bloom_level": {
                                "type": "string",
                                "enum": ["remember", "understand", "apply", "analyse", "evaluate", "create"],
                            },
                            "question_text": {"type": "string", "minLength": 1},
                            "options": {"type": ["array", "null"], "items": {"type": "string"}},
                            "correct_answer": {"type": "string", "minLength": 1},
                            "acceptable_answers": {"type": ["array", "null"], "items": {"type": "string"}},
                            "explanation": {"type": "string", "minLength": 1},
                            "marks": {"type": "integer", "minimum": 1, "maximum": 5},
                            "rubric": {"type": ["string", "null"]},
                            "answer_space": {
                                "type": "string",
                                "enum": ["multiple_choice", "single_line", "short_paragraph", "long_paragraph"],
                            },
                            "source_reference": {"type": ["string", "null"]},
                            "image_slot": {"type": "null"},
                        },
                        "required": [
                            "slot",
                            "question_type",
                            "difficulty",
                            "bloom_level",
                            "question_text",
                            "options",
                            "correct_answer",
                            "acceptable_answers",
                            "explanation",
                            "marks",
                            "rubric",
                            "answer_space",
                            "source_reference",
                            "image_slot",
                        ],
                    },
                },
            },
        }

        base_messages = [
            {"role": "system", "content": "Return only valid JSON matching the required schema. No markdown fences, no prose."},
            {"role": "user", "content": prompt},
        ]
        retry_instruction = (
            "Your previous response was not valid JSON or did not match the required schema. "
            "Fix only the JSON formatting/structure and field values needed for schema compliance; "
            "do not change the educational content beyond what is necessary. "
            "Return ONLY the JSON array."
        )

        structured = await call_openrouter_structured_json(
            model=_QUIZ_MODEL,
            base_messages=base_messages,
            response_format=response_format,
            temperature_first=0.15,
            temperature_retry=0.0,
            retry_user_instruction=retry_instruction,
            validator=_validate_questions,
            logger=logger,
            log_context=f"[generate_quiz][{session_id}]",
        )
        await asyncio.to_thread(_touch_session, {"generation_stage": "parsing_response"})
        questions = structured.value
        for att in structured.attempts:
            usage = AIUsage(
                prompt_tokens=att.llm_result.usage.prompt_tokens,
                completion_tokens=att.llm_result.usage.completion_tokens,
                total_tokens=att.llm_result.usage.total_tokens,
                model=att.llm_result.usage.model,
                request_id=att.llm_result.usage.request_id,
            )
            await record_ai_usage(
                svc,
                user_id=teacher_id,
                source="quiz_builder",
                entity_type="quiz_session",
                entity_id=session_id,
                route="job:generate_quiz",
                usage=usage,
                status="success" if att.status == "success" else "error",
                error_message=None if att.status == "success" else (att.error_message or att.status),
            )
    except StructuredJSONError as exc:
        logger.error("[generate_quiz] structured output failed session=%s job_id=%s err=%s", session_id, job_id, exc)
        for att in getattr(exc, "attempts", []) or []:
            usage = AIUsage(
                prompt_tokens=att.llm_result.usage.prompt_tokens,
                completion_tokens=att.llm_result.usage.completion_tokens,
                total_tokens=att.llm_result.usage.total_tokens,
                model=att.llm_result.usage.model,
                request_id=att.llm_result.usage.request_id,
            )
            await record_ai_usage(
                svc,
                user_id=teacher_id,
                source="quiz_builder",
                entity_type="quiz_session",
                entity_id=session_id,
                route="job:generate_quiz",
                usage=usage,
                status="error",
                error_message=att.error_message or att.status,
            )
        internal_msg = str(getattr(exc, "last_exc", None) or exc)[:300]
        if not await asyncio.to_thread(_still_current_job):
            logger.info("[generate_quiz] superseded before failing session=%s job_id=%s", session_id, job_id)
            return
        await asyncio.to_thread(_touch_session, {
            "status": "error",
            "error_message": internal_msg or "LLM structured output failed",
            "generation_status": "failed",
            "generation_stage": "failed",
            "generation_failed_at": _now_iso(),
            "generation_error_code": "structured_output_error",
            "generation_error_message": internal_msg or "structured_output_error",
        })
        return
    except Exception as exc:
        logger.error("[generate_quiz] generation failed session=%s job_id=%s exc=%s", session_id, job_id, exc)
        await record_ai_usage(
            svc,
            user_id=teacher_id,
            source="quiz_builder",
            entity_type="quiz_session",
            entity_id=session_id,
            route="job:generate_quiz",
            status="error",
            error_message=str(exc),
        )
        if not await asyncio.to_thread(_still_current_job):
            logger.info("[generate_quiz] superseded before failing session=%s job_id=%s", session_id, job_id)
            return
        await asyncio.to_thread(_touch_session, {
            "status": "error",
            "error_message": "LLM service error",
            "generation_status": "failed",
            "generation_stage": "failed",
            "generation_failed_at": _now_iso(),
            "generation_error_code": "llm_service_error",
            "generation_error_message": str(exc)[:300],
        })
        return

    def _save(qs: list[dict[str, Any]]) -> None:
        if not _still_current_job():
            logger.info("[generate_quiz] superseded before saving questions session=%s job_id=%s", session_id, job_id)
            return
        _touch_session({"generation_stage": "saving_questions"})
        if not _still_current_job():
            logger.info("[generate_quiz] superseded before question delete/insert session=%s job_id=%s", session_id, job_id)
            return
        svc.table("quiz_questions").delete().eq("session_id", session_id).execute()
        if qs:
            svc.table("quiz_questions").insert([
                {
                    "session_id": session_id,
                    "slot": q.get("slot", i + 1),
                    "question_json": q,
                    "validation_status": "pending",
                    "validation_warnings": [],
                    "teacher_approved": False,
                }
                for i, q in enumerate(qs)
            ]).execute()
        if not _still_current_job():
            logger.info("[generate_quiz] superseded before completing session=%s job_id=%s", session_id, job_id)
            return
        _touch_session({
            "status": "review",
            "generation_status": "completed",
            "generation_stage": "completed",
            "generation_completed_at": _now_iso(),
        })

    await asyncio.to_thread(_save, questions)
    logger.info("[generate_quiz] generation completed session=%s job_id=%s questions=%d", session_id, job_id, len(questions))
