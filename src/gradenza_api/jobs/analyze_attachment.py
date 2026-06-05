"""
ARQ job: analyze_attachment (PROMPT7 / PROMPT8)

Downloads, extracts, and AI-summarises an Online Lesson attachment.
Updates online_lesson_attachments row from 'processing' → 'ready' or 'error'.

Run by:
  arq gradenza_api.worker.WorkerSettings
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from typing import Any

from gradenza_api.services.attachment_extraction import ExtractionResult, extract
from gradenza_api.services.openrouter import call_openrouter, strip_json_fences
from gradenza_api.services.supabase_client import get_service_client
from gradenza_api.services.usage import AIUsage, record_ai_usage

logger = logging.getLogger(__name__)

# Max characters of extracted text forwarded to the summary model
_MAX_SUMMARY_INPUT = 8_000

# Max characters stored in analysis_json.extracted_text (for AI context)
_MAX_CONTEXT_TEXT = 8_000

_SUMMARY_SYSTEM = """\
You are an educational content analyst. Analyse the extracted content from a teacher-uploaded file
and return a structured educational summary as JSON.

Output a JSON object with exactly these fields — no extras, no omissions:
{
  "summary": "<2–4 sentence educational summary: what topics and skills this file covers and how a teacher might use it>",
  "detected_subject": "<subject name, e.g. Mathematics, Physics, Chemistry — or empty string>",
  "detected_exam_system": "<exam system, e.g. IB AA HL, GCSE, SAT, EGE, WAEC — or empty string>",
  "detected_level": "<student level, e.g. Year 11, IB HL, Grade 10 AP, A-Level — or empty string>",
  "main_topics": ["<specific topic 1>", "<specific topic 2>"],
  "learning_objectives": ["<specific learning outcome 1>", "<specific learning outcome 2>"],
  "key_definitions": ["<Term: concise definition>"],
  "worked_examples": ["<brief description of a worked example found in the file>"],
  "practice_question_candidates": ["<brief description of a question or problem that could become a practice question>"],
  "misconceptions": ["<common misconception or error pattern this material addresses or implies>"],
  "vocabulary": ["<key term 1>", "<key term 2>"],
  "diagrams_or_visuals": ["<description of a diagram or visual element and its educational purpose — include page/slide reference if available>"],
  "recommended_lesson_uses": ["<specific way a teacher could use this file, e.g. 'Use slide 3 as a starter problem on integration'>"]
}

Guidelines:
- For images: describe visual elements under diagrams_or_visuals.
- For PDFs/PPTX: include page or slide references where possible (e.g. "Slide 4 worked example").
- Keep each array item concise (1–2 sentences max).
- Leave arrays empty [] if no relevant content found — never omit a field.
- Return only valid JSON. No markdown fences, no preamble."""


# ── Job entry point ────────────────────────────────────────────────────────────

async def analyze_attachment(
    ctx: dict,
    *,
    attachment_id: str,
    lesson_id: str,
    user_id: str,
) -> dict[str, Any]:
    """
    ARQ job that downloads, extracts, and summarises an attachment.

    ctx: ARQ context (not used directly; required by ARQ protocol)
    attachment_id: online_lesson_attachments.id
    lesson_id: parent lessons.id
    user_id: teacher's user ID (for AI usage logging)
    """
    svc = get_service_client()
    logger.info(
        "[analyze-attachment] start attachment=%s lesson=%s user=%s",
        attachment_id, lesson_id, user_id,
    )

    # ── Fetch attachment row ─────────────────────────────────────────────────

    def _fetch() -> dict | None:
        return (
            svc.table("online_lesson_attachments")
            .select(
                "id, lesson_id, teacher_id, storage_bucket, storage_path, "
                "mime_type, file_type, file_name"
            )
            .eq("id", attachment_id)
            .eq("lesson_id", lesson_id)
            .maybe_single()
            .execute()
            .data
        )

    attachment = await asyncio.to_thread(functools.partial(_fetch))
    if not attachment:
        logger.error("[analyze-attachment] attachment %s not found", attachment_id)
        return {"error": "Attachment not found"}

    storage_bucket = attachment.get("storage_bucket") or "material-files"
    storage_path = attachment.get("storage_path")
    mime_type = attachment.get("mime_type") or ""
    file_type = attachment.get("file_type") or ""
    file_name = attachment.get("file_name") or "attachment"

    if not storage_path:
        await _set_error(svc, attachment_id, "No storage path recorded for this attachment.")
        return {"error": "No storage path"}

    # ── Download file ────────────────────────────────────────────────────────

    try:
        def _download() -> bytes:
            return svc.storage.from_(storage_bucket).download(storage_path)

        file_bytes: bytes = await asyncio.to_thread(functools.partial(_download))
    except Exception as exc:
        logger.error("[analyze-attachment] download failed: %s", exc)
        await _set_error(svc, attachment_id, f"Could not download file: {exc}")
        return {"error": str(exc)}

    if not file_bytes:
        await _set_error(svc, attachment_id, "Downloaded file was empty.")
        return {"error": "Empty file"}

    # ── Extract content ──────────────────────────────────────────────────────

    extraction: ExtractionResult
    ocr_usage: AIUsage | None = None

    extraction, ocr_usage = await extract(
        file_bytes,
        mime_type,
        file_type,
        svc=svc,
        user_id=user_id,
        entity_id=attachment_id,
    )

    if extraction.error:
        await _set_error(svc, attachment_id, extraction.error)
        return {"error": extraction.error}

    # needs_ocr=True here means the file had sparse pages AND OCR was never
    # attempted (e.g. rasterizer unavailable).  If we also have no text at all
    # the file is genuinely unreadable — hard-error.  If we have partial text,
    # continue to ready so the teacher sees what was extracted plus the warning.
    if extraction.needs_ocr and not extraction.extracted_text.strip():
        await _set_error(
            svc,
            attachment_id,
            "Scanned PDF: no extractable text was found and OCR could not be "
            "attempted. Try a text-based PDF or upload individual pages as images.",
        )
        return {"error": "No extractable content in scanned PDF"}

    if ocr_usage:
        await record_ai_usage(
            svc,
            user_id=user_id,
            source="online_lesson_attachment_ocr",
            entity_type="attachment",
            entity_id=attachment_id,
            route="ARQ/analyze_attachment",
            usage=ocr_usage,
            status="success",
        )

    # ── Generate educational summary via AI ──────────────────────────────────

    summary = ""
    analysis_json: dict[str, Any] = {}
    summary_usage: AIUsage | None = None

    if extraction.extracted_text.strip():
        content_preview = extraction.extracted_text[:_MAX_SUMMARY_INPUT]
        if len(extraction.extracted_text) > _MAX_SUMMARY_INPUT:
            content_preview += "\n\n[... content truncated for analysis ...]"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": f"File name: {file_name}\n\nContent:\n{content_preview}",
            },
        ]

        try:
            result = await call_openrouter(
                model="google/gemini-2.5-flash",
                messages=messages,
                temperature=0.2,
            )
            summary_usage = AIUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                model=result.usage.model,
                request_id=result.usage.request_id,
            )
            raw = strip_json_fences(result.content or "")
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    summary = str(parsed.pop("summary", "")).strip()
                    analysis_json = parsed
            except (json.JSONDecodeError, ValueError):
                logger.warning("[analyze-attachment] summary JSON parse failed; using raw")
                summary = raw[:500]

        except Exception as exc:
            logger.warning("[analyze-attachment] summary AI call failed: %s", exc)
            summary = (
                f"Content extracted ({extraction.token_estimate or 0} estimated tokens). "
                "AI summary unavailable — check API connectivity."
            )

    if summary_usage:
        await record_ai_usage(
            svc,
            user_id=user_id,
            source="online_lesson_attachment_summary",
            entity_type="attachment",
            entity_id=attachment_id,
            route="ARQ/analyze_attachment",
            usage=summary_usage,
            status="success",
        )

    # ── Compose analysis_json ────────────────────────────────────────────────
    # Store a safe excerpt of extracted_text here so the frontend can build
    # prompt context without a separate round-trip to fetch the full column.

    if extraction.warnings:
        analysis_json["warnings"] = extraction.warnings

    if extraction.extracted_text:
        analysis_json["extracted_text"] = extraction.extracted_text[:_MAX_CONTEXT_TEXT]

    # ── Persist to DB ────────────────────────────────────────────────────────

    def _update() -> None:
        svc.table("online_lesson_attachments").update(
            {
                "status": "ready",
                "extracted_text": extraction.extracted_text or None,
                "summary": summary or None,
                "analysis_json": analysis_json,
                "page_count": extraction.page_count,
                "slide_count": extraction.slide_count,
                "image_count": extraction.image_count,
                "token_estimate": extraction.token_estimate,
                "error_message": None,
            }
        ).eq("id", attachment_id).execute()

    try:
        await asyncio.to_thread(functools.partial(_update))
    except Exception as exc:
        logger.error("[analyze-attachment] DB update failed: %s", exc)
        await _set_error(svc, attachment_id, f"DB update failed: {exc}")
        return {"error": str(exc)}

    logger.info(
        "[analyze-attachment] done attachment=%s tokens=%s pages=%s slides=%s",
        attachment_id,
        extraction.token_estimate,
        extraction.page_count,
        extraction.slide_count,
    )

    return {
        "status": "ready",
        "token_estimate": extraction.token_estimate,
        "page_count": extraction.page_count,
        "slide_count": extraction.slide_count,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _set_error(svc: Any, attachment_id: str, message: str) -> None:
    """Update attachment status to 'error' (best-effort, swallows exceptions)."""
    def _update() -> None:
        try:
            svc.table("online_lesson_attachments").update(
                {
                    "status": "error",
                    "error_message": message[:1000],
                }
            ).eq("id", attachment_id).execute()
        except Exception as exc:
            logger.error("[analyze-attachment] could not set error status: %s", exc)

    await asyncio.to_thread(functools.partial(_update))
