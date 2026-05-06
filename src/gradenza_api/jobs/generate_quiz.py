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
from typing import Any

from gradenza_api.services.openrouter import call_openrouter, strip_json_fences
from gradenza_api.services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

_QUIZ_MODEL = "google/gemini-2.5-flash"

_GENERATE_PROMPT = """\
You are an expert quiz creator. Generate high-quality quiz questions.

Session context:
- Subject: {subject}
- Grade level: {grade_level}
- Mode: {mode}
- Preset: {preset}

Source material:
{source_text}

Blueprint — generate EXACTLY these questions:
{blueprint}

Return a JSON array where each element follows this schema:
{{
  "slot": <integer matching blueprint slot>,
  "question_type": "multiple_choice" | "fill_blank" | "short_answer" | "long_answer",
  "difficulty": "easy" | "medium" | "hard",
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
    logger.info("[generate_quiz] job started session=%s", session_id)
    svc = get_service_client()

    def _set_generating() -> None:
        svc.table("quiz_builder_sessions").update({"status": "generating", "error_message": None}).eq("id", session_id).execute()

    await asyncio.to_thread(_set_generating)

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
            .select("blueprint_json")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ).data
        bp = bp[0] if bp else None

        return sess, src, bp

    session, source, blueprint = await asyncio.to_thread(_fetch)

    if not session:
        logger.error("[generate_quiz] session not found session=%s", session_id)
        return

    source_text: str = (source or {}).get("extracted_text") or "(no source text provided)"
    blueprint_items: list[Any] = (blueprint or {}).get("blueprint_json") or []

    prompt = _GENERATE_PROMPT.format(
        subject=session.get("subject") or "General",
        grade_level=session.get("grade_level") or "Secondary",
        mode=session.get("mode", "from_prompt"),
        preset=session.get("preset", "standard_worksheet"),
        source_text=source_text[:3000],
        blueprint=json.dumps(blueprint_items, indent=2),
    )

    def _set_error(msg: str) -> None:
        svc.table("quiz_builder_sessions").update({
            "status": "error",
            "error_message": msg,
        }).eq("id", session_id).execute()

    try:
        result = await call_openrouter(
            model=_QUIZ_MODEL,
            temperature=0.4,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = strip_json_fences(result.content)
        questions: list[dict[str, Any]] = json.loads(raw)
        if not isinstance(questions, list):
            raise ValueError(f"Expected list, got {type(questions).__name__}")
    except Exception as exc:
        logger.error("[generate_quiz] generation failed session=%s exc=%s", session_id, exc)
        await asyncio.to_thread(_set_error, str(exc))
        return

    def _save(qs: list[dict[str, Any]]) -> None:
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
        svc.table("quiz_builder_sessions").update({"status": "review"}).eq("id", session_id).execute()

    await asyncio.to_thread(_save, questions)
    logger.info("[generate_quiz] generation completed session=%s questions=%d", session_id, len(questions))
