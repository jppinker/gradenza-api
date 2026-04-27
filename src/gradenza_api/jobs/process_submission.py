"""
ARQ job: process_submission(ctx, submission_id, assignment_question_id=None, has_assignment_question_id=False, force=False)

Orchestrates OCR + grading for one submission:
  1. Fetch all submission_photos rows.
  2. For each unprocessed photo: download from Storage → OCR → write results.
  3. When all photos done: advance submissions.status → 'ocr_done'.
  4. Grade via LLM (one question if scoped; otherwise legacy full-assignment).
  5. Advance submissions.status → 'graded'.

Ported from:
  - supabase/functions/ocr-process/index.ts
  - supabase/functions/grade-submission/index.ts

Error reporting
───────────────
Terminal failures write a machine-readable code to submissions.processing_error
(+ processing_error_at timestamp) so the frontend can stop polling instead of
hanging on "Processing…".  The field is cleared to NULL at job start so
re-queued jobs always begin from a clean state.

  Code                      When set
  ────────────────────────  ─────────────────────────────────────────────────
  photo_fetch_error         DB exception while querying submission_photos
  no_photos                 No rows found after PHOTO_FETCH_ATTEMPTS retries
  storage_download_error    Supabase Storage download returned None/failed
  ocr_incomplete            Not all photos have ocr_done_at after OCR loop
  assignment_question_id mismatch  Submission/question scoping mismatch (safety abort)
  submission_not_found      submissions row missing when refetched for grading
  no_grading_prompt         No active grading_prompt_versions for this exam system
  no_assignment_questions   assignment_questions empty for this assignment
  no_resolvable_questions   All question rows failed to resolve from source table
  db_write_error            One or more grading_results upserts failed

Status enum: draft | submitted | ocr_done | graded | reviewed  (no "failed").
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from gradenza_api.services.openrouter import call_openrouter, strip_json_fences
from gradenza_api.services.supabase_client import get_service_client
from gradenza_api.services.usage import AIUsage, record_ai_usage
from gradenza_api.settings import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PHOTO_FETCH_ATTEMPTS = 3
PHOTO_FETCH_RETRY_DELAY = 2.0  # seconds between retries

OCR_MODEL = "google/gemini-2.5-flash"
OCR_TEMPERATURE = 0.1
OCR_LOW_CONFIDENCE = 0.60

GRADING_MODEL = "google/gemini-2.5-flash"
GRADING_TEMPERATURE = 0.1
LLM_LOW_CONFIDENCE = 0.72

CROSSED_OUT_PATTERN = re.compile(r"\[CROSSED OUT:", re.IGNORECASE)

OCR_SYSTEM_PROMPT = """\
You are an OCR engine for student handwritten mathematics exam submissions.
Your task is to accurately extract ALL written content from the image provided.

Output a single JSON object with exactly two fields:
  "text": a string with the complete extracted content
  "confidence": a float 0.0–1.0 reflecting OCR quality

Rules for "text":
- Preserve question-part labels exactly as written: (a), (b), (c), etc., each on its own line
- Render mathematical expressions in LaTeX notation (e.g. \\frac{1}{2}, x^2)
- If handwriting is crossed out, include it as [CROSSED OUT: <content>]
- If there is a drawn diagram, include [DIAGRAM] at the relevant position
- Separate question parts with a blank line
- Do not paraphrase or interpret — transcribe verbatim

Rules for "confidence":
  1.0  = all text clearly legible, no ambiguity
  0.8–0.99 = mostly legible, minor uncertain characters
  0.6–0.79 = legible but some words/symbols uncertain
  0.4–0.59 = significant uncertainty; portions may be wrong
  0.0–0.39 = large portions unreadable or image quality very poor

Respond ONLY with valid JSON. No markdown fences, no preamble, no explanation."""


# ── OCR helpers ───────────────────────────────────────────────────────────────

def _clamp01(n: float) -> float:
    return max(0.0, min(1.0, n))


def _mime_for_path(path: str) -> str:
    """Return the MIME type inferred from a storage-path file extension."""
    if path.endswith(".pdf"):
        return "application/pdf"
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


async def _fetch_photos_retrying(
    fetch_fn: Callable[[], list[dict]],
    submission_id: str,
    *,
    max_attempts: int = PHOTO_FETCH_ATTEMPTS,
    retry_delay: float = PHOTO_FETCH_RETRY_DELAY,
) -> list[dict]:
    """Fetch submission photos, retrying on empty results or transient errors.

    Handles eventual consistency: Supabase Storage triggers can insert
    submission_photos rows slightly after the job is enqueued, so an empty
    result on the first attempt is not necessarily a hard failure.

    Returns the first non-empty list, or [] after all attempts are exhausted.
    A fetch exception is logged and retried; on the final attempt it propagates.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            photos = await asyncio.to_thread(fetch_fn)
        except Exception as exc:
            if attempt == max_attempts:
                raise
            logger.warning(
                "[job] photo fetch error submission=%s attempt=%d/%d: %s — retrying in %.0fs",
                submission_id, attempt, max_attempts, exc, retry_delay,
            )
            await asyncio.sleep(retry_delay)
            continue

        if photos:
            return photos

        if attempt < max_attempts:
            logger.warning(
                "[job] no photos yet submission=%s attempt=%d/%d — retrying in %.0fs",
                submission_id, attempt, max_attempts, retry_delay,
            )
            await asyncio.sleep(retry_delay)

    return []


async def _run_ocr(
    image_base64: str,
    mime_type: str,
    *,
    user_id: str,
    submission_id: str,
) -> tuple[str, float]:
    """Run OCR via OpenRouter. Returns (text, confidence). Records usage event."""
    messages = [
        {"role": "system", "content": OCR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                },
                {
                    "type": "text",
                    "text": "Extract all written content from this student exam submission page.",
                },
            ],
        },
    ]

    svc = get_service_client()
    try:
        result = await call_openrouter(
            model=OCR_MODEL,
            temperature=OCR_TEMPERATURE,
            messages=messages,
        )
        raw = result.content
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
    except Exception as exc:
        logger.error("[ocr] OpenRouter call failed: %s", exc)
        await record_ai_usage(
            svc,
            user_id=user_id,
            source="autograde",
            entity_type="submission",
            entity_id=submission_id,
            route="job:process_submission/ocr",
            status="error",
            error_message=str(exc),
        )
        return "", 0.0

    await record_ai_usage(
        svc,
        user_id=user_id,
        source="autograde",
        entity_type="submission",
        entity_id=submission_id,
        route="job:process_submission/ocr",
        usage=ai_usage,
        status="success",
    )

    stripped = strip_json_fences(raw)
    try:
        parsed = json.loads(stripped)
        text = parsed.get("text", raw) if isinstance(parsed.get("text"), str) else raw
        conf = _clamp01(float(parsed["confidence"])) if isinstance(parsed.get("confidence"), (int, float)) else 0.5
        return text, conf
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("[ocr] JSON parse failed; using raw content")
        return raw, 0.5


# ── Grading data structures ───────────────────────────────────────────────────

@dataclass
class ParsedPart:
    label: str
    marks: int
    text: str | None


@dataclass
class ParsedMarkschemeStep:
    part_label: str | None
    description: str
    marks: int
    mark_type: str | None  # 'M', 'A', 'R', 'AG', 'FT', or None


@dataclass
class ParsedQuestion:
    question_uuid: str
    source_id: str
    problem_text: str
    diagram_required: bool
    parts: list[ParsedPart]
    markscheme_steps: list[ParsedMarkschemeStep]
    marks_available: int
    ft_eligible_parts: list[str]
    ft_dependencies: dict[str, str]
    assignment_question_id: str


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_parts(raw: Any) -> list[ParsedPart]:
    if not isinstance(raw, list) or not raw:
        return [ParsedPart(label="a", marks=0, text=None)]
    result = []
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            continue
        label = (p.get("label") or p.get("part") or chr(ord("a") + i)).lower()
        marks = int(p.get("marks") or p.get("mark") or 0)
        text = p.get("text") or p.get("problem_text") or p.get("content") or None
        result.append(ParsedPart(label=label, marks=marks, text=text))
    return result or [ParsedPart(label="a", marks=0, text=None)]


_VALID_MARK_TYPES = {"M", "A", "R", "AG", "FT"}


def _parse_markscheme_steps(raw: Any) -> list[ParsedMarkschemeStep]:
    if not isinstance(raw, list) or not raw:
        return []
    result = []
    for s in raw:
        if not isinstance(s, dict):
            continue
        raw_type = str(s.get("mark_type") or s.get("type") or "").upper().strip()
        mark_type = raw_type if raw_type in _VALID_MARK_TYPES else None
        part_label_raw = s.get("part") or s.get("part_label") or s.get("label")
        part_label = str(part_label_raw).lower() if part_label_raw else None
        description = str(s.get("description") or s.get("text") or s.get("content") or "")
        marks = int(s.get("marks") or s.get("mark") or 0)
        result.append(
            ParsedMarkschemeStep(
                part_label=part_label,
                description=description,
                marks=marks,
                mark_type=mark_type,
            )
        )
    return result


def _infer_ft_fields(
    steps: list[ParsedMarkschemeStep],
) -> tuple[list[str], dict[str, str]]:
    """Infer FT-eligible parts and dependencies from markscheme steps."""
    ft_parts: list[str] = []
    ft_deps: dict[str, str] = {}

    part_order: list[str] = []
    for s in steps:
        if s.part_label and s.part_label not in part_order:
            part_order.append(s.part_label)

    for s in steps:
        if s.mark_type == "FT" and s.part_label and s.part_label not in ft_parts:
            ft_parts.append(s.part_label)
            idx = part_order.index(s.part_label)
            if idx > 0:
                ft_deps[s.part_label] = part_order[idx - 1]

    return ft_parts, ft_deps


def _total_available_marks(
    steps: list[ParsedMarkschemeStep], parts: list[ParsedPart]
) -> int:
    from_steps = sum(s.marks for s in steps if s.mark_type != "AG")
    if from_steps > 0:
        return from_steps
    return sum(p.marks for p in parts)


# ── LLM prompt builders ───────────────────────────────────────────────────────

def _build_markscheme_text(steps: list[ParsedMarkschemeStep]) -> str:
    if not steps:
        return "(No markscheme steps available)"
    lines: list[str] = []
    current_part = ""
    for step in steps:
        part_header = f"Part ({step.part_label})" if step.part_label else "General"
        if part_header != current_part:
            if current_part:
                lines.append("")
            lines.append(f"### {part_header}")
            current_part = part_header
        type_label = (
            f"[{step.mark_type}{step.marks}]"
            if step.mark_type
            else f"[{step.marks} mark{'s' if step.marks != 1 else ''}]"
        )
        lines.append(f"{type_label} {step.description}")
    return "\n".join(lines)


def _build_parts_text(parts: list[ParsedPart]) -> str:
    if not parts:
        return "(No parts defined)"
    if len(parts) == 1 and not parts[0].text:
        return f"Single part — {parts[0].marks} mark{'s' if parts[0].marks != 1 else ''} available"
    return "\n".join(
        f"({p.label}) [{p.marks} marks] {p.text or '(shares question stem)'}"
        for p in parts
    )


def _build_ft_context_text(
    ft_eligible_parts: list[str],
    ft_dependencies: dict[str, str],
    extracted_answers: dict[str, str | None],
) -> str:
    if not ft_eligible_parts:
        return "(No FT-eligible parts in this question)"
    lines = ["Follow-Through (FT) information:"]
    for part in ft_eligible_parts:
        dep = ft_dependencies.get(part)
        extracted = extracted_answers.get(dep) if dep else None
        if dep:
            lines.append(
                f"- Part ({part}) is FT-eligible; depends on part ({dep})."
                + (
                    f" Student's extracted answer from part ({dep}): {extracted}"
                    if extracted
                    else f" Student's answer from part ({dep}) not yet extracted — apply FT if a value is visible in the OCR."
                )
            )
        else:
            lines.append(
                f"- Part ({part}) is FT-eligible (no specific dependency recorded)."
            )
    return "\n".join(lines)


# ── LLM grading call ──────────────────────────────────────────────────────────

async def _call_grading_llm(
    *,
    system_prompt: str,
    question: ParsedQuestion,
    ocr_text: str,
    has_low_confidence_ocr: bool,
    extracted_answers: dict[str, str | None],
    user_id: str,
    submission_id: str,
) -> dict | None:
    note_low_ocr = (
        "\n**Note for examiner:** One or more pages in this submission had OCR confidence "
        "below threshold. Some text may be inaccurately transcribed. Apply amber flag if "
        "any uncertainty arises from potential OCR error.\n"
        if has_low_confidence_ocr
        else ""
    )

    diagram_note = (
        "**Note:** This question requires a diagram as part of the answer. "
        "Apply informational label only — do not withhold marks solely for diagram quality.\n\n"
        if question.diagram_required
        else ""
    )

    user_message = f"""## Question

{question.problem_text}

### Parts
{_build_parts_text(question.parts)}

### Marks available: {question.marks_available}

{diagram_note}---

## Markscheme

{_build_markscheme_text(question.markscheme_steps)}

---

## Follow-Through (FT) Context

{_build_ft_context_text(question.ft_eligible_parts, question.ft_dependencies, extracted_answers)}

---

## Student's OCR-transcribed Work

{ocr_text.strip() or '(No OCR text available — student may not have answered this question)'}

{note_low_ocr}---

Grade this question following the IB marking conventions in the system prompt. Return valid JSON only."""

    svc = get_service_client()
    try:
        result = await call_openrouter(
            model=GRADING_MODEL,
            temperature=GRADING_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        raw = result.content
        ai_usage = AIUsage(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            model=result.usage.model,
            request_id=result.usage.request_id,
        )
    except Exception as exc:
        logger.error("[grade] OpenRouter call failed: %s", exc)
        await record_ai_usage(
            svc,
            user_id=user_id,
            source="autograde",
            entity_type="submission",
            entity_id=submission_id,
            route="job:process_submission/grade",
            status="error",
            error_message=str(exc),
        )
        return None

    await record_ai_usage(
        svc,
        user_id=user_id,
        source="autograde",
        entity_type="submission",
        entity_id=submission_id,
        route="job:process_submission/grade",
        usage=ai_usage,
        status="success",
    )

    stripped = strip_json_fences(raw)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("[grade] JSON parse failed; raw preview: %s", raw[:200])
        return None


# ── Amber + trust layer helpers ───────────────────────────────────────────────

def _detect_amber(ocr_text: str, llm_result: dict | None) -> tuple[bool, str | None]:
    if CROSSED_OUT_PATTERN.search(ocr_text):
        return True, "Crossed-out or overwritten work detected in OCR transcription"
    if llm_result is None:
        return True, "Grading LLM call failed — manual review required"
    confidence = float(llm_result.get("confidence") or 1.0)
    if confidence < LLM_LOW_CONFIDENCE:
        return True, f"LLM confidence {confidence * 100:.0f}% below threshold — uncertain grading"
    return False, None


def _classify_trust_layer(
    amber_flag: bool, marks_awarded: int, marks_available: int
) -> str:
    if amber_flag:
        return "layer2_amber"
    if marks_awarded == 0 and marks_available > 0:
        return "layer3_incorrect"
    return "layer1_auto"


# ── Main job function ─────────────────────────────────────────────────────────

async def process_submission(
    ctx: dict,
    submission_id: str,
    assignment_question_id: str | None = None,
    has_assignment_question_id: bool = False,
    force: bool = False,
) -> dict:
    """
    ARQ job: orchestrate OCR + grading for one submission.
    ctx is the ARQ job context (we don't use any special context values here).
    """
    # Backward compatibility: older enqueues passed (submission_id, force)
    if isinstance(assignment_question_id, bool) and has_assignment_question_id is False:
        force = assignment_question_id
        assignment_question_id = None
        has_assignment_question_id = False

    logger.info(
        "[job] start submission=%s force=%s assignment_question_id=%s has_aqid=%s",
        submission_id,
        force,
        assignment_question_id,
        has_assignment_question_id,
    )
    svc = get_service_client()

    # ── Error helpers (closures over svc + submission_id) ──────────────────────
    def _clear_processing_error() -> None:
        try:
            svc.table("submissions").update({
                "processing_error": None,
                "processing_error_at": None,
            }).eq("id", submission_id).execute()
        except Exception as exc:
            logger.warning(
                "[job] submission=%s: could not clear processing_error: %s", submission_id, exc
            )

    def _set_processing_error(code: str) -> None:
        logger.error("[job] submission=%s processing_error=%s", submission_id, code)
        try:
            svc.table("submissions").update({
                "processing_error": code,
                "processing_error_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", submission_id).execute()
        except Exception as exc:
            logger.error(
                "[job] submission=%s: could not write processing_error=%s: %s",
                submission_id, code, exc,
            )

    # Clear any error left by a previous attempt so the frontend stops showing it.
    await asyncio.to_thread(_clear_processing_error)

    # ── Safety: validate submission ↔ AQID scoping ───────────────────────────
    def _fetch_submission_scope() -> dict | None:
        res = (
            svc.table("submissions")
            .select("id, status, student_id, assignment_id, assignment_question_id")
            .eq("id", submission_id)
            .maybe_single()
            .execute()
        )
        return res.data

    submission_scope = await asyncio.to_thread(_fetch_submission_scope)
    if not submission_scope:
        await asyncio.to_thread(_set_processing_error, "submission_not_found")
        return {"submission_id": submission_id, "error": "submission_not_found"}

    # user_id for all usage events in this job: the student who owns the submission
    job_user_id: str = submission_scope.get("student_id") or ""

    submission_aqid = submission_scope.get("assignment_question_id")
    if has_assignment_question_id:
        if submission_aqid != assignment_question_id:
            await asyncio.to_thread(_set_processing_error, "assignment_question_id mismatch")
            return {"submission_id": submission_id, "error": "assignment_question_id mismatch"}
    else:
        assignment_question_id = submission_aqid

    is_question_scoped = submission_aqid is not None

    # ── 1. Fetch submission_photos (with retry for eventual consistency) ─────────
    # submission_photos rows may arrive slightly after the job is enqueued (the
    # Storage trigger is asynchronous), so we retry a few times before giving up.
    def _fetch_photos() -> list[dict]:
        res = (
            svc.table("submission_photos")
            .select("id, submission_id, question_id, page_number, storage_path, ocr_done_at")
            .eq("submission_id", submission_id)
            .order("page_number")
            .execute()
        )
        return res.data or []

    try:
        photos = await _fetch_photos_retrying(_fetch_photos, submission_id)
    except Exception as exc:
        logger.error(
            "[job] submission=%s: photo fetch failed after %d attempts: %s",
            submission_id, PHOTO_FETCH_ATTEMPTS, exc,
        )
        await asyncio.to_thread(_set_processing_error, "photo_fetch_error")
        return {"submission_id": submission_id, "error": "photo_fetch_error"}

    if not photos:
        logger.error(
            "[job] submission=%s: no photos after %d attempts — upload may not have completed",
            submission_id, PHOTO_FETCH_ATTEMPTS,
        )
        await asyncio.to_thread(_set_processing_error, "no_photos")
        return {"submission_id": submission_id, "error": "no_photos"}

    # ── Safety: ensure fetched photos are correctly scoped ─────────
    wrong_submission_ids = {
        p.get("submission_id")
        for p in photos
        if p.get("submission_id") and p.get("submission_id") != submission_id
    }
    if wrong_submission_ids:
        await asyncio.to_thread(_set_processing_error, "assignment_question_id mismatch")
        return {"submission_id": submission_id, "error": "assignment_question_id mismatch"}

    if is_question_scoped:
        wrong_photo_aqids = {
            p.get("question_id")
            for p in photos
            if p.get("question_id") not in (None, submission_aqid)
        }
        if wrong_photo_aqids:
            await asyncio.to_thread(_set_processing_error, "assignment_question_id mismatch")
            return {"submission_id": submission_id, "error": "assignment_question_id mismatch"}

    # ── 2. OCR each unprocessed photo ─────────────────────────────
    for photo in photos:
        if photo.get("ocr_done_at") and not force:
            logger.debug("[job] photo %s already processed — skip", photo["id"])
            continue

        storage_path: str = photo["storage_path"]
        bucket: str = settings.submission_photos_bucket

        # Download from Supabase Storage
        def _download(b: str, p: str) -> bytes | None:
            try:
                return svc.storage.from_(b).download(p)
            except Exception as exc:
                logger.error("[job] storage download failed bucket=%s path=%s: %s", b, p, exc)
                return None

        raw_bytes = await asyncio.to_thread(_download, bucket, storage_path)

        if raw_bytes is None:
            await asyncio.to_thread(_set_processing_error, "storage_download_error")
            return {"submission_id": submission_id, "error": "storage_download_error"}

        mime_type = _mime_for_path(storage_path)

        image_base64 = base64.b64encode(raw_bytes).decode("utf-8")

        ocr_text, ocr_confidence = await _run_ocr(
            image_base64,
            mime_type,
            user_id=job_user_id,
            submission_id=submission_id,
        )

        # Write OCR result
        def _write_ocr(photo_id: str, text: str, conf: float) -> None:
            svc.table("submission_photos").update({
                "ocr_text": text,
                "ocr_confidence": conf,
                "ocr_done_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", photo_id).execute()

        await asyncio.to_thread(_write_ocr, photo["id"], ocr_text, ocr_confidence)
        logger.info(
            "[job] OCR done photo=%s page=%s confidence=%.2f",
            photo["id"],
            photo.get("page_number"),
            ocr_confidence,
        )

    # ── 3. Check if all pages are done ────────────────────────────
    def _check_all_done() -> list[dict]:
        res = (
            svc.table("submission_photos")
            .select("ocr_done_at")
            .eq("submission_id", submission_id)
            .execute()
        )
        return res.data or []

    all_photos = await asyncio.to_thread(_check_all_done)
    all_done = bool(all_photos) and all(p.get("ocr_done_at") for p in all_photos)

    if not all_done:
        logger.error("[job] submission=%s: not all photos OCR'd", submission_id)
        await asyncio.to_thread(_set_processing_error, "ocr_incomplete")
        return {"submission_id": submission_id, "error": "ocr_incomplete"}

    # Advance status from 'submitted' → 'ocr_done' (idempotent)
    def _advance_to_ocr_done() -> None:
        svc.table("submissions").update({"status": "ocr_done"}).eq(
            "id", submission_id
        ).eq("status", "submitted").execute()

    await asyncio.to_thread(_advance_to_ocr_done)

    # ── 4. Grading ────────────────────────────────────────────────

    # 4a. Fetch submission + assignment
    def _fetch_submission() -> dict | None:
        res = (
            svc.table("submissions")
            .select(
                "id, status, student_id, assignment_id, assignment_question_id, "
                "assignments(id, grading_style, class_id, classes(id, exam_system_id, exam_systems(id, code)))"
            )
            .eq("id", submission_id)
            .maybe_single()
            .execute()
        )
        return res.data

    submission_row = await asyncio.to_thread(_fetch_submission)
    if not submission_row:
        logger.error("[job] submission=%s not found", submission_id)
        await asyncio.to_thread(_set_processing_error, "submission_not_found")
        return {"submission_id": submission_id, "error": "submission_not_found"}

    if submission_row.get("status") != "ocr_done":
        logger.warning(
            "[job] submission %s status=%s — skip grading",
            submission_id,
            submission_row.get("status"),
        )
        return {"submission_id": submission_id, "skipped": True, "reason": f"status is {submission_row.get('status')}"}

    # Refresh user_id in case student_id was missing from the initial scope fetch
    if not job_user_id:
        job_user_id = submission_row.get("student_id") or ""

    submission_aqid = submission_row.get("assignment_question_id")
    if has_assignment_question_id:
        if submission_aqid != assignment_question_id:
            await asyncio.to_thread(_set_processing_error, "assignment_question_id mismatch")
            return {"submission_id": submission_id, "error": "assignment_question_id mismatch"}
    else:
        assignment_question_id = submission_aqid

    is_question_scoped = submission_aqid is not None

    assignment = submission_row.get("assignments") or {}
    if isinstance(assignment, list):
        assignment = assignment[0] if assignment else {}
    grading_style: str = assignment.get("grading_style") or "ib_style"

    classes = assignment.get("classes") or {}
    if isinstance(classes, list):
        classes = classes[0] if classes else {}
    exam_systems = classes.get("exam_systems") or {}
    if isinstance(exam_systems, list):
        exam_systems = exam_systems[0] if exam_systems else {}

    exam_system_code: str = exam_systems.get("code") or "IB"
    exam_system_id: str = classes.get("exam_system_id") or ""

    # 4b. Fetch active grading prompt
    def _fetch_prompt() -> dict | None:
        res = (
            svc.table("grading_prompt_versions")
            .select("id, prompt_text, version_tag")
            .eq("exam_system_id", exam_system_id)
            .eq("is_active", True)
            .maybe_single()
            .execute()
        )
        return res.data

    prompt_row = await asyncio.to_thread(_fetch_prompt)
    if not prompt_row:
        logger.error(
            "[job] submission=%s: no active grading prompt for exam_system_id=%s",
            submission_id, exam_system_id,
        )
        await asyncio.to_thread(_set_processing_error, "no_grading_prompt")
        return {"submission_id": submission_id, "error": "no_grading_prompt"}

    system_prompt: str = prompt_row["prompt_text"]
    prompt_version_id: str = prompt_row["id"]

    # 4c. Fetch all OCR text
    def _fetch_ocr_pages() -> list[dict]:
        res = (
            svc.table("submission_photos")
            .select("submission_id, question_id, page_number, ocr_text, ocr_confidence")
            .eq("submission_id", submission_id)
            .order("page_number")
            .execute()
        )
        return res.data or []

    ocr_pages = await asyncio.to_thread(_fetch_ocr_pages)

    wrong_ocr_submission_ids = {
        p.get("submission_id")
        for p in ocr_pages
        if p.get("submission_id") and p.get("submission_id") != submission_id
    }
    if wrong_ocr_submission_ids:
        await asyncio.to_thread(_set_processing_error, "assignment_question_id mismatch")
        return {"submission_id": submission_id, "error": "assignment_question_id mismatch"}

    if is_question_scoped:
        wrong_ocr_aqids = {
            p.get("question_id")
            for p in ocr_pages
            if p.get("question_id") not in (None, submission_aqid)
        }
        if wrong_ocr_aqids:
            await asyncio.to_thread(_set_processing_error, "assignment_question_id mismatch")
            return {"submission_id": submission_id, "error": "assignment_question_id mismatch"}

    full_ocr_text = "\n\n--- Page break ---\n\n".join(
        p["ocr_text"] for p in ocr_pages if p.get("ocr_text")
    )
    has_low_confidence_ocr = any(
        p.get("ocr_confidence") is not None and float(p["ocr_confidence"]) < OCR_LOW_CONFIDENCE
        for p in ocr_pages
    )

    # 4d. Fetch assignment questions
    def _fetch_aq_rows() -> list[dict]:
        q = (
            svc.table("assignment_questions")
            .select(
                "id, assignment_id, position, question_id, "
                "questions(id, source_id, source_table, diagram_required, ft_eligible_parts, ft_dependencies)"
            )
        )

        if is_question_scoped:
            res = (
                q.eq("id", submission_aqid)
                .eq("assignment_id", assignment.get("id"))
                .execute()
            )
        else:
            res = (
                q.eq("assignment_id", assignment.get("id"))
                .order("position")
                .execute()
            )

        return res.data or []

    aq_rows = await asyncio.to_thread(_fetch_aq_rows)
    if not aq_rows:
        if is_question_scoped:
            logger.error(
                "[job] submission=%s: assignment_question_id=%s not found for assignment=%s",
                submission_id,
                submission_aqid,
                assignment.get("id"),
            )
            await asyncio.to_thread(_set_processing_error, "assignment_question_id mismatch")
            return {"submission_id": submission_id, "error": "assignment_question_id mismatch"}

        logger.error("[job] submission=%s: no assignment_questions rows", submission_id)
        await asyncio.to_thread(_set_processing_error, "no_assignment_questions")
        return {"submission_id": submission_id, "error": "no_assignment_questions"}

    # 4e. Resolve IB question details for each assignment question
    questions: list[ParsedQuestion] = []

    for aq in aq_rows:
        q_row = aq.get("questions") or {}
        if isinstance(q_row, list):
            q_row = q_row[0] if q_row else {}
        if not q_row:
            logger.warning("[job] no question row for aq.id=%s", aq.get("id"))
            continue

        source_table: str = q_row.get("source_table") or "ib_math_questionbank"
        source_id: str = str(q_row.get("source_id") or "")
        if not source_id:
            logger.warning("[job] question %s has no source_id", q_row.get("id"))
            continue

        def _fetch_ib_row(sid: str) -> dict | None:
            if source_table == "ib_math_questionbank":
                res = (
                    svc.table("ib_math_questionbank")
                    .select(
                        "id, domain, theme, theme_slug, level, difficulty, problem_text, "
                        "question_media_count, parts_count, parts_json, "
                        "markscheme_steps_count, markscheme_steps_json"
                    )
                    .eq("id", int(sid))
                    .maybe_single()
                    .execute()
                )
                return res.data
            return None

        raw_row = await asyncio.to_thread(_fetch_ib_row, source_id)
        if not raw_row:
            logger.warning(
                "[job] source row not found for source_id=%s table=%s", source_id, source_table
            )
            continue

        parts = _parse_parts(raw_row.get("parts_json"))
        markscheme_steps = _parse_markscheme_steps(raw_row.get("markscheme_steps_json"))
        marks_available = _total_available_marks(markscheme_steps, parts)

        # FT metadata: prefer questions table values, fall back to inference
        ft_from_q_parts = q_row.get("ft_eligible_parts") or []
        ft_from_q_deps = q_row.get("ft_dependencies") or {}

        if ft_from_q_parts:
            ft_eligible = [str(p) for p in ft_from_q_parts]
            ft_deps = {str(k): str(v) for k, v in ft_from_q_deps.items()}
        else:
            ft_eligible, ft_deps = _infer_ft_fields(markscheme_steps)

        questions.append(
            ParsedQuestion(
                question_uuid=str(q_row.get("id")),
                source_id=source_id,
                problem_text=str(raw_row.get("problem_text") or ""),
                diagram_required=(raw_row.get("question_media_count") or 0) > 0
                or bool(q_row.get("diagram_required")),
                parts=parts,
                markscheme_steps=markscheme_steps,
                marks_available=marks_available,
                ft_eligible_parts=ft_eligible,
                ft_dependencies=ft_deps,
                assignment_question_id=str(aq["id"]),
            )
        )

    if not questions:
        logger.error("[job] submission=%s: no resolvable questions", submission_id)
        await asyncio.to_thread(_set_processing_error, "no_resolvable_questions")
        return {"submission_id": submission_id, "error": "no_resolvable_questions"}

    if is_question_scoped:
        if len(questions) != 1 or questions[0].assignment_question_id != str(submission_aqid):
            logger.error(
                "[job] submission=%s: scoped assignment_question_id mismatch expected=%s got=%s",
                submission_id,
                submission_aqid,
                [q.assignment_question_id for q in questions],
            )
            await asyncio.to_thread(_set_processing_error, "assignment_question_id mismatch")
            return {"submission_id": submission_id, "error": "assignment_question_id mismatch"}

    # ── 5. Grade each question ─────────────────────────────────────
    graded_count = 0
    error_count = 0
    is_ib_style = grading_style == "ib_style"

    for question in questions:
        extracted_answers: dict[str, str | None] = {}

        if is_ib_style:
            llm_result = await _call_grading_llm(
                system_prompt=system_prompt,
                question=question,
                ocr_text=full_ocr_text,
                has_low_confidence_ocr=has_low_confidence_ocr,
                extracted_answers=extracted_answers,
                user_id=job_user_id,
                submission_id=submission_id,
            )
        else:
            simple_q = ParsedQuestion(
                question_uuid=question.question_uuid,
                source_id=question.source_id,
                problem_text=question.problem_text,
                diagram_required=question.diagram_required,
                parts=question.parts,
                markscheme_steps=question.markscheme_steps,
                marks_available=question.marks_available,
                ft_eligible_parts=[],
                ft_dependencies={},
                assignment_question_id=question.assignment_question_id,
            )
            llm_result = await _call_grading_llm(
                system_prompt=system_prompt
                + "\n\nThis assignment uses SIMPLE grading (correct/incorrect + explanation only). "
                "Do not apply IB-style mark type distinctions or FT logic.",
                question=simple_q,
                ocr_text=full_ocr_text,
                has_low_confidence_ocr=has_low_confidence_ocr,
                extracted_answers=extracted_answers,
                user_id=job_user_id,
                submission_id=submission_id,
            )

        # Populate extracted_answers from LLM result
        if llm_result and isinstance(llm_result.get("parts"), list):
            for part in llm_result["parts"]:
                pl = part.get("part_label")
                ea = part.get("extracted_answer")
                if pl and ea:
                    extracted_answers[pl] = ea

        # Amber + trust layer
        extra_amber, extra_amber_reason = _detect_amber(full_ocr_text, llm_result)
        final_amber = extra_amber or bool(llm_result and llm_result.get("overall_amber_flag")) or has_low_confidence_ocr
        final_amber_reason = (
            extra_amber_reason
            or (
                "Low OCR confidence on one or more submission pages"
                if has_low_confidence_ocr
                else None
            )
            or (llm_result.get("overall_amber_reason") if llm_result else None)
        )

        marks_awarded = int(llm_result.get("total_marks_awarded") or 0) if llm_result else 0
        marks_available_q = question.marks_available
        confidence = float(llm_result.get("confidence") or 0) if llm_result else 0.0
        feedback_text = str(llm_result.get("feedback_text") or "") if llm_result else ""
        ft_applied = any(
            p.get("ft_applied") for p in (llm_result.get("parts") or [])
        ) if llm_result else False

        llm_assessment = (
            json.dumps({
                "full_assessment": llm_result.get("full_assessment"),
                "parts": llm_result.get("parts"),
            })
            if llm_result
            else '{"error":"LLM call failed"}'
        )

        trust_layer = _classify_trust_layer(final_amber, marks_awarded, marks_available_q)

        def _upsert_result(aq_id: str) -> bool:
            try:
                svc.table("grading_results").upsert(
                    {
                        "submission_id": submission_id,
                        "assignment_question_id": aq_id,
                        "prompt_version_id": prompt_version_id,
                        "ocr_text": full_ocr_text,
                        "llm_assessment": llm_assessment,
                        "marks_awarded": marks_awarded,
                        "marks_available": marks_available_q,
                        "method_marks_awarded": (
                            llm_result.get("total_method_marks") if llm_result else None
                        ),
                        "accuracy_marks_awarded": (
                            llm_result.get("total_accuracy_marks") if llm_result else None
                        ),
                        "ft_marks_awarded": (
                            llm_result.get("total_ft_marks") if llm_result else None
                        ),
                        "ft_applied": ft_applied,
                        "feedback_text": feedback_text,
                        "confidence_score": confidence,
                        "trust_layer": trust_layer,
                        "amber_flag": final_amber,
                        "amber_reason": final_amber_reason,
                        "diagram_flag": question.diagram_required,
                        "teacher_overridden": False,
                        "student_flagged": False,
                    },
                    on_conflict="submission_id,assignment_question_id",
                ).execute()
                return True
            except Exception as exc:
                logger.error("[job] grading_results upsert failed aq=%s: %s", aq_id, exc)
                return False

        ok = await asyncio.to_thread(_upsert_result, question.assignment_question_id)
        if ok:
            graded_count += 1
            logger.info(
                "[job] graded aq=%s marks=%d/%d amber=%s trust=%s",
                question.assignment_question_id,
                marks_awarded,
                marks_available_q,
                final_amber,
                trust_layer,
            )
        else:
            error_count += 1

    # ── 6. Advance status to 'graded' ─────────────────────────────
    if error_count > 0:
        logger.error(
            "[job] submission=%s: %d grading_results upsert(s) failed — status stays 'ocr_done'",
            submission_id, error_count,
        )
        await asyncio.to_thread(_set_processing_error, "db_write_error")
    elif graded_count > 0:
        def _advance_to_graded() -> None:
            svc.table("submissions").update({
                "status": "graded",
                "graded_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", submission_id).eq("status", "ocr_done").execute()

        await asyncio.to_thread(_advance_to_graded)

    logger.info(
        "[job] complete submission=%s graded=%d/%d errors=%d prompt=%s",
        submission_id,
        graded_count,
        len(questions),
        error_count,
        prompt_row.get("version_tag"),
    )

    return {
        "submission_id": submission_id,
        "questions_graded": graded_count,
        "questions_total": len(questions),
        "errors": error_count,
        "prompt_version": prompt_row.get("version_tag"),
    }
