"""
Shared extraction service for Online Lesson attachments (PROMPT6).

Handles:
  - PDF text extraction via pypdf with OCR fallback for scanned pages (PROMPT20)
  - PPTX slide extraction via python-pptx with tables and alt-text (PROMPT21)
  - Image OCR/vision via OpenRouter Gemini 2.5 Flash (async)

Each extractor returns an ExtractionResult dataclass with normalised
counts, truncated text, and warnings so callers don't need to know
which backend was used.

OCR fallback strategy for sparse/scanned PDF pages (PROMPT20 fix):
  1. Try the cheap path first: pull the embedded raster image out of the page
     (works for JPEG/PNG/WEBP encoded pages — no decode needed).
  2. If no readable raster is found (CCITT fax, JBIG2, JPEG2000, vector PDF,
     etc.) fall back to pypdfium2 + Pillow rasterization.  pypdfium2 (PDFium,
     BSD/Apache-2.0) renders the page to an RGB bitmap; Pillow encodes it as
     JPEG so it can be forwarded to the vision model.
  3. If pypdfium2/Pillow are unavailable, degrade gracefully: partial text is
     kept and a warning is added rather than hard-erroring the whole file.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gradenza_api.services.usage import AIUsage

logger = logging.getLogger(__name__)

# Characters-per-token approximation for budget estimation
_CHARS_PER_TOKEN = 4

# Maximum characters stored in extracted_text (DB column limit safety)
_MAX_EXTRACTED_TEXT = 100_000

# Maximum characters sent to the vision model per image
_MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB sanity guard

# PDF limits and sparse-detection thresholds (PROMPT20)
_MAX_PDF_FILE_SIZE: int = 30 * 1024 * 1024   # 30 MB
_MAX_PDF_PAGES: int = 50                      # pages processed per file
_SPARSE_CHARS_PER_PAGE: int = 50             # chars/page below this → likely scanned
_MAX_OCR_PAGES: int = 5                       # max pages to OCR (cost / latency control)
_MIN_EMBEDDED_IMAGE_BYTES: int = 20_000      # skip tiny decorative images

# Rasterization settings
_RASTER_DPI: int = 200
_RASTER_JPEG_QUALITY: int = 85

# PPTX limits (PROMPT21)
_MAX_PPTX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

_VISION_MODEL = "google/gemini-2.5-flash"
_VISION_TEMPERATURE = 0.1


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    extracted_text: str = ""
    page_count: int | None = None
    slide_count: int | None = None
    image_count: int | None = None
    token_estimate: int | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    needs_ocr: bool = False  # True only if OCR was needed but NEVER attempted

    def _estimate_tokens(self) -> None:
        if self.extracted_text:
            self.token_estimate = max(1, len(self.extracted_text) // _CHARS_PER_TOKEN)

    def _truncate(self) -> None:
        if len(self.extracted_text) > _MAX_EXTRACTED_TEXT:
            self.extracted_text = self.extracted_text[:_MAX_EXTRACTED_TEXT]
            self.warnings.append("Extracted text was truncated at 100 000 characters.")


# ── PDF extraction ─────────────────────────────────────────────────────────────

def _extract_pdf_pages(file_bytes: bytes) -> tuple[ExtractionResult, list[int]]:
    """
    Extract text page-by-page via pypdf.  Returns the result and the list of
    0-indexed page numbers whose text is below the sparse threshold.

    Determining sparseness per-page (rather than by document average) prevents
    silent content loss in mixed text+scanned PDFs.
    """
    result = ExtractionResult()

    if len(file_bytes) > _MAX_PDF_FILE_SIZE:
        size_mb = len(file_bytes) / (1024 * 1024)
        result.error = f"PDF file is too large ({size_mb:.1f} MB). Maximum allowed is 30 MB."
        return result, []

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(file_bytes))
        result.page_count = len(reader.pages)

        pages_to_process = min(result.page_count, _MAX_PDF_PAGES)
        if result.page_count > _MAX_PDF_PAGES:
            result.warnings.append(
                f"PDF has {result.page_count} pages — only the first {_MAX_PDF_PAGES} were processed."
            )

        pages: list[str] = []
        sparse_page_indices: list[int] = []  # 0-indexed

        for i in range(pages_to_process):
            page = reader.pages[i]
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[Page {i + 1}]\n{text}")
            if len(text) < _SPARSE_CHARS_PER_PAGE:
                sparse_page_indices.append(i)

        result.extracted_text = "\n\n".join(pages)

        if sparse_page_indices:
            result.needs_ocr = True  # will be cleared once OCR is attempted

    except Exception as exc:
        logger.warning("[_extract_pdf_pages] failed: %s", exc)
        result.error = f"PDF extraction failed: {exc}"
        return result, []

    result._truncate()
    result._estimate_tokens()
    return result, sparse_page_indices


def extract_pdf(file_bytes: bytes) -> ExtractionResult:
    """
    Sync PDF text extraction (pypdf only, no OCR).  Used by callers that don't
    need the async fallback path.  Sets result.needs_ocr when sparse pages
    are detected.
    """
    result, _ = _extract_pdf_pages(file_bytes)
    return result


# ── PDF OCR helpers ────────────────────────────────────────────────────────────

def _guess_image_mime(data: bytes) -> str | None:
    """Guess image MIME type from magic bytes."""
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _extract_embedded_raster(file_bytes: bytes, page_index: int) -> tuple[bytes, str] | None:
    """
    Try to pull a directly-readable raster image from a single PDF page (cheap
    path: no rendering required).  Returns (image_bytes, mime_type) or None.

    Only works when the page stores a JPEG/PNG/WEBP object that pypdf can
    return as raw bytes.  CCITT, JBIG2, JPEG2000, and multi-image composites
    all fall through to None so the caller can use the rasterization path.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        if page_index >= len(reader.pages):
            return None
        for img_file in reader.pages[page_index].images:
            try:
                data = img_file.data
                if not data or len(data) < _MIN_EMBEDDED_IMAGE_BYTES:
                    continue
                mime = _guess_image_mime(data)
                if mime:
                    return data, mime
            except Exception:
                continue
    except Exception:
        pass
    return None


def _rasterize_pdf_pages(
    file_bytes: bytes,
    page_indices: list[int],
    *,
    dpi: int = _RASTER_DPI,
) -> list[tuple[int, bytes, str]]:
    """
    Render the given 0-indexed PDF pages to JPEG using pypdfium2 + Pillow.
    Returns list of (1-indexed page num, jpeg_bytes, "image/jpeg").

    Respects _MAX_OCR_PAGES.  Returns [] gracefully if pypdfium2 or Pillow
    are not installed, so the rest of extraction continues without crashing.
    """
    try:
        import pypdfium2 as pdfium  # type: ignore[import-untyped]
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError:
        return []

    results: list[tuple[int, bytes, str]] = []
    try:
        doc = pdfium.PdfDocument(file_bytes)
    except Exception as exc:
        logger.warning("[_rasterize_pdf_pages] could not open PDF: %s", exc)
        return []

    try:
        scale = dpi / 72.0  # PDFium works in points (72 pt/inch)
        for page_index in page_indices:
            if len(results) >= _MAX_OCR_PAGES:
                break
            if page_index >= len(doc):
                continue
            try:
                page = doc[page_index]
                bitmap = page.render(scale=scale, rotation=0)
                pil_img: Image.Image = bitmap.to_pil()
                # Convert RGBA/P/CMYK → RGB before JPEG encoding
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=_RASTER_JPEG_QUALITY, optimize=True)
                jpeg_bytes = buf.getvalue()
                if jpeg_bytes and len(jpeg_bytes) <= _MAX_IMAGE_BYTES:
                    results.append((page_index + 1, jpeg_bytes, "image/jpeg"))
                elif jpeg_bytes:
                    logger.warning(
                        "[_rasterize_pdf_pages] page %d JPEG is %d bytes — skipping",
                        page_index + 1,
                        len(jpeg_bytes),
                    )
            except Exception as exc:
                logger.warning(
                    "[_rasterize_pdf_pages] could not render page %d: %s", page_index + 1, exc
                )
    finally:
        doc.close()

    return results


async def extract_pdf_with_fallback(
    file_bytes: bytes,
    *,
    svc: Any,
    user_id: str,
    entity_id: str,
) -> tuple["ExtractionResult", "AIUsage | None"]:
    """
    PDF extraction with per-page OCR fallback for sparse/scanned pages (PROMPT20).

    Strategy per sparse page:
      1. Try embedded raster (cheap — no rendering cost).
      2. If none found, rasterize via pypdfium2 + Pillow.
    Each image is OCR'd via extract_image_via_vision.

    needs_ocr is set to False once OCR has been attempted (whether or not it
    produced text), so downstream code can distinguish "tried, got nothing"
    from "never tried".  A partial result (some text from some pages) is
    always returned as status=ready with warnings rather than hard-failing.
    """
    from gradenza_api.services.usage import AIUsage

    result, sparse_indices = await asyncio.to_thread(_extract_pdf_pages, file_bytes)

    if result.error:
        return result, None

    if not sparse_indices:
        # All pages had sufficient text; nothing to OCR
        result.needs_ocr = False
        return result, None

    # ── Collect per-sparse-page image sources ────────────────────────────────
    # For each sparse page, prefer a directly-readable embedded raster; fall
    # back to rasterization for pages where that returns nothing.
    pages_needing_render: list[int] = []
    page_images: list[tuple[int, bytes, str]] = []  # (1-indexed page, bytes, mime)

    for idx in sparse_indices:
        if len(page_images) >= _MAX_OCR_PAGES:
            break
        embedded = await asyncio.to_thread(_extract_embedded_raster, file_bytes, idx)
        if embedded:
            page_images.append((idx + 1, embedded[0], embedded[1]))
        else:
            pages_needing_render.append(idx)

    # Rasterize remaining sparse pages in one call (sync, in thread)
    remaining_budget = _MAX_OCR_PAGES - len(page_images)
    if pages_needing_render and remaining_budget > 0:
        render_targets = pages_needing_render[:remaining_budget]
        rendered = await asyncio.to_thread(
            _rasterize_pdf_pages, file_bytes, render_targets
        )
        page_images.extend(rendered)

    # Mark OCR as attempted regardless of whether images were found
    result.needs_ocr = False

    if not page_images:
        # Sparse pages exist but no images could be obtained (no embedded raster
        # and pypdfium2/Pillow not available).  Keep whatever text pypdf found
        # and add an informative warning — do NOT hard-error.
        skipped = len(sparse_indices)
        result.warnings.append(
            f"{skipped} page(s) appear scanned or image-based and could not be "
            "read via OCR (PDF rasterizer unavailable). "
            "For best results, upload a text-based PDF or individual page images."
        )
        result._estimate_tokens()
        return result, None

    # ── OCR each image ───────────────────────────────────────────────────────
    ocr_parts: list[str] = []
    total_prompt = 0
    total_completion = 0
    total_total = 0
    last_model = ""
    last_request_id = ""

    for page_num, img_bytes, mime_type in page_images:
        try:
            ocr_result, usage = await extract_image_via_vision(
                img_bytes, mime_type, svc=svc, user_id=user_id, entity_id=entity_id
            )
            if ocr_result.extracted_text.strip():
                ocr_parts.append(
                    f"[Page {page_num} — OCR]\n{ocr_result.extracted_text.strip()}"
                )
            if usage:
                total_prompt += usage.prompt_tokens or 0
                total_completion += usage.completion_tokens or 0
                total_total += usage.total_tokens or 0
                last_model = usage.model or last_model
                last_request_id = usage.request_id or last_request_id
        except Exception as exc:
            logger.warning(
                "[extract_pdf_with_fallback] OCR failed for page %d: %s", page_num, exc
            )

    # ── Merge OCR output into result ─────────────────────────────────────────
    if ocr_parts:
        combined = "\n\n".join(filter(None, [result.extracted_text, *ocr_parts]))
        result.extracted_text = combined[:_MAX_EXTRACTED_TEXT]
        # Replace vague sparse warning with concrete OCR success note
        result.warnings = [
            w for w in result.warnings
            if "low text density" not in w.lower() and "no extractable text" not in w.lower()
            and "scanned or image-based" not in w.lower()
        ]
        result.warnings.append(
            f"OCR fallback applied to {len(ocr_parts)} page(s) using vision model."
        )
        # Warn if some sparse pages produced no OCR text
        unread = len(page_images) - len(ocr_parts)
        if unread > 0:
            result.warnings.append(
                f"{unread} scanned page(s) could not be read via OCR — "
                "the content may be illegible or a non-educational image."
            )
    else:
        # OCR was attempted but produced no text on any page
        skipped_but_attempted = len(page_images)
        result.warnings.append(
            f"OCR was attempted on {skipped_but_attempted} scanned page(s) "
            "but no text could be extracted. "
            "The pages may be blank, illegible, or contain only decorative images."
        )

    result._estimate_tokens()

    if total_total > 0:
        combined_usage = AIUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_total,
            model=last_model,
            request_id=last_request_id,
        )
        return result, combined_usage

    return result, None


# ── PPTX extraction ────────────────────────────────────────────────────────────

def extract_pptx(file_bytes: bytes) -> ExtractionResult:
    """
    Extract slide text from a PPTX using python-pptx.  Sync — safe to run in a thread.

    Enhancements (PROMPT21):
    - File size limit enforced before parsing.
    - Legacy .ppt files (binary format) produce a clear error message.
    - Table cells are extracted row-by-row with pipe separators.
    - Picture alt-text (descr attribute) is included where present.
    """
    result = ExtractionResult()

    # ── Size limit ──────────────────────────────────────────────────────────
    if len(file_bytes) > _MAX_PPTX_FILE_SIZE:
        size_mb = len(file_bytes) / (1024 * 1024)
        result.error = (
            f"PowerPoint file is too large ({size_mb:.1f} MB). Maximum allowed is 50 MB."
        )
        return result

    # ── Import guard ────────────────────────────────────────────────────────
    try:
        from pptx import Presentation  # type: ignore[import-untyped]
        from pptx.enum.shapes import MSO_SHAPE_TYPE  # type: ignore[import-untyped]
    except ImportError as exc:
        result.error = f"python-pptx is not installed: {exc}"
        return result

    # ── Parse file — catch legacy PPT early ─────────────────────────────────
    try:
        prs = Presentation(io.BytesIO(file_bytes))
    except Exception as exc:
        err_lower = str(exc).lower()
        if any(kw in err_lower for kw in ("bad zip", "not a zip", "zipfile", "zip file")):
            result.error = (
                "This appears to be a legacy .ppt file. "
                "Please convert it to .pptx format and upload again."
            )
        else:
            result.error = f"PPTX extraction failed: {exc}"
        return result

    # ── Content extraction ───────────────────────────────────────────────────
    try:
        result.slide_count = len(prs.slides)
        slides: list[str] = []

        for i, slide in enumerate(prs.slides, 1):
            parts: list[str] = []

            # Slide title
            if slide.shapes.title and slide.shapes.title.text.strip():
                parts.append(f"Title: {slide.shapes.title.text.strip()}")

            for shape in slide.shapes:
                # Skip title (already handled above)
                if slide.shapes.title and shape == slide.shapes.title:
                    continue

                # Text frames (text boxes, placeholders, etc.)
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            parts.append(text)

                # Tables — extract cell text row by row
                elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                    try:
                        table = shape.table
                        table_rows: list[str] = []
                        for row in table.rows:
                            cells = [
                                cell.text.strip()
                                for cell in row.cells
                                if cell.text.strip()
                            ]
                            if cells:
                                table_rows.append(" | ".join(cells))
                        if table_rows:
                            parts.append("[Table]\n" + "\n".join(table_rows))
                    except Exception:
                        pass

                # Pictures — include alt-text (descr attribute) if present
                elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    try:
                        nv_pr = getattr(shape.element, "nvPicPr", None)
                        if nv_pr is not None:
                            cnv_pr = getattr(nv_pr, "cNvPr", None)
                            if cnv_pr is not None:
                                alt_text = cnv_pr.get("descr", "").strip()
                                if alt_text:
                                    parts.append(f"[Image: {alt_text}]")
                    except Exception:
                        pass

            # Speaker notes
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"[Notes: {notes}]")

            if parts:
                slides.append(f"[Slide {i}]\n" + "\n".join(parts))

        result.extracted_text = "\n\n".join(slides)

        if not result.extracted_text.strip():
            result.warnings.append("PPTX contained no extractable text.")

    except Exception as exc:
        logger.warning("[extract_pptx] content extraction failed: %s", exc)
        result.error = f"PPTX content extraction failed: {exc}"
        return result

    result._truncate()
    result._estimate_tokens()
    return result


# ── Image vision extraction ────────────────────────────────────────────────────

_VISION_SYSTEM = """\
You are analysing an educational image (worksheet, slide, diagram, graph, whiteboard photo, or textbook scan).

Extract ALL visible content with educational precision across these categories:

1. VISIBLE TEXT: Transcribe all text exactly as written, preserving structure (headings, numbered lists, bullets).
2. MATHEMATICS: Render all math notation in LaTeX.
   - Inline: $\\frac{a}{b}$, $x^{2}+y^{2}=r^{2}$, $\\sin\\theta$
   - Block: $$\\int_{a}^{b} f(x)\\,dx$$
   - Include every formula, equation, expression, and numerical value visible.
3. DIAGRAMS & GRAPHS: Describe axes labels, units, curve shapes, intersection points, and key features.
4. QUESTIONS & PROBLEMS: Extract any question statements, sub-parts (a), (b), (c), and mark allocations verbatim.
5. EDUCATIONAL CONTEXT: Identify likely subject, exam system, and topic from content cues.

Output a JSON object with exactly these fields:
{
  "text": "<all visible text, math (in LaTeX), and descriptions as coherent plain text — complete, nothing omitted>",
  "math_expressions": ["<LaTeX expr 1>", "<LaTeX expr 2>"],
  "diagram_description": "<detailed description of charts, graphs, or diagrams visible, or empty string>",
  "question_statements": ["<verbatim question or sub-question 1>", "<verbatim question 2>"],
  "likely_topic": "<inferred topic/concept, e.g. 'Quadratic equations by completing the square', or empty string>",
  "likely_subject": "<inferred subject, e.g. 'Mathematics', or empty string>",
  "confidence": <float 0.0–1.0 — 1.0 = all text clear, 0.0 = unreadable>,
  "image_description": "<one sentence describing what the image shows overall>",
  "educationally_useful": <true if the image contains educationally useful content, false otherwise>
}

Return only valid JSON. No markdown fences, no preamble."""


async def extract_image_via_vision(
    file_bytes: bytes,
    mime_type: str,
    *,
    svc: Any,
    user_id: str,
    entity_id: str,
) -> tuple[ExtractionResult, "AIUsage | None"]:
    """
    Extract text and content from an image using Gemini 2.5 Flash vision.
    Returns (ExtractionResult, AIUsage|None).  Async.
    """
    from gradenza_api.services.openrouter import call_openrouter, strip_json_fences
    from gradenza_api.services.usage import AIUsage

    result = ExtractionResult(image_count=1)

    if len(file_bytes) > _MAX_IMAGE_BYTES:
        result.error = "Image file exceeds 10 MB limit."
        return result, None

    image_b64 = base64.b64encode(file_bytes).decode("utf-8")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _VISION_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                },
                {"type": "text", "text": "Extract all content from this educational image."},
            ],
        },
    ]

    try:
        api_result = await call_openrouter(
            model=_VISION_MODEL,
            temperature=_VISION_TEMPERATURE,
            messages=messages,
        )
        usage = AIUsage(
            prompt_tokens=api_result.usage.prompt_tokens,
            completion_tokens=api_result.usage.completion_tokens,
            total_tokens=api_result.usage.total_tokens,
            model=api_result.usage.model,
            request_id=api_result.usage.request_id,
        )
        raw = strip_json_fences(api_result.content or "")
        try:
            parsed = json.loads(raw)
            text = str(parsed.get("text") or "").strip()
            desc = str(parsed.get("image_description") or "").strip()
            diagram_desc = str(parsed.get("diagram_description") or "").strip()
            confidence = float(parsed.get("confidence") or 0.5)
            question_stmts: list[str] = [
                str(q) for q in (parsed.get("question_statements") or []) if q
            ]

            # Build enriched extracted_text with all relevant sections
            sections: list[str] = [text] if text else []
            if diagram_desc:
                sections.append(f"[Diagram: {diagram_desc}]")
            if question_stmts:
                qs_block = "\n".join(f"  - {q}" for q in question_stmts)
                sections.append(f"[Questions identified:\n{qs_block}]")
            if desc and desc not in text:
                sections.append(f"[Image: {desc}]")

            result.extracted_text = "\n\n".join(s for s in sections if s).strip()

            if confidence < 0.6:
                result.warnings.append(
                    f"Low OCR confidence ({confidence:.2f}) — some text may be inaccurate."
                )
            if not parsed.get("educationally_useful", True):
                result.warnings.append(
                    "Image may not contain educationally useful content — consider uploading a clearer file."
                )
        except (json.JSONDecodeError, ValueError):
            result.extracted_text = raw.strip()
            result.warnings.append("Could not parse structured vision response — using raw output.")

    except Exception as exc:
        logger.error("[extract_image_via_vision] OpenRouter error: %s", exc)
        result.error = f"Image OCR failed: {exc}"
        return result, None

    result._truncate()
    result._estimate_tokens()
    return result, usage


# ── Dispatch helper ────────────────────────────────────────────────────────────

async def extract(
    file_bytes: bytes,
    mime_type: str,
    file_type: str,
    *,
    svc: Any,
    user_id: str,
    entity_id: str,
) -> tuple[ExtractionResult, "AIUsage | None"]:
    """
    Dispatch to the correct extractor based on MIME / file_type.
    Always returns (ExtractionResult, usage).

    PDFs use extract_pdf_with_fallback which attempts vision OCR on sparse pages.
    PPTX uses extract_pptx (sync, no AI calls).
    Images use extract_image_via_vision (async, AI call).
    """
    if file_type == "pdf" or mime_type == "application/pdf":
        return await extract_pdf_with_fallback(
            file_bytes, svc=svc, user_id=user_id, entity_id=entity_id
        )

    if file_type == "pptx" or "presentationml" in mime_type:
        result = await asyncio.to_thread(extract_pptx, file_bytes)
        return result, None

    if file_type == "image" or mime_type.startswith("image/"):
        return await extract_image_via_vision(
            file_bytes, mime_type, svc=svc, user_id=user_id, entity_id=entity_id
        )

    result = ExtractionResult(error=f"Unsupported MIME type: {mime_type}")
    return result, None
