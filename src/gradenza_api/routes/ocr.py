"""
POST /v1/ocr/image-to-text

Converts a base64-encoded image to extracted text via
OpenRouter → google/gemini-2.5-flash (vision).

Ported from supabase/functions/ocr-image-to-text/index.ts.
Now also returns `confidence` (the edge function parsed it but discarded it).
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Annotated

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import call_openrouter, strip_json_fences

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/ocr", tags=["ocr"])

# ── Constants ──────────────────────────────────────────────────────────────────

OCR_MODEL = "google/gemini-2.5-flash"
OCR_TEMPERATURE = 0.1

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/heic",
}

OCR_SYSTEM_PROMPT = """\
You are an accurate OCR engine for educational documents and materials.
Extract all text content from the provided image verbatim.

Output a single JSON object with exactly two fields:
  "text": the complete extracted text as a plain string
  "confidence": a float 0.0–1.0 reflecting OCR quality

Rules for "text":
- Transcribe all visible text exactly as written
- Preserve paragraph structure with newlines
- Render mathematical expressions in LaTeX (e.g. \\frac{1}{2})
- For tables, use a plain text representation with spacing
- Include headings and labels as written
- If a section is unreadable, write [UNREADABLE]

Rules for "confidence":
  1.0  = all text clearly legible
  0.8  = mostly legible, minor uncertain characters
  0.6  = some uncertain words or symbols
  0.4  = significant uncertainty
  0.0  = largely unreadable

Respond ONLY with valid JSON. No markdown fences, no preamble."""


# ── Request / Response models ──────────────────────────────────────────────────

class OcrRequest(BaseModel):
    imageBase64: str = Field(..., min_length=1)
    mimeType: str = Field(default="image/jpeg")


class OcrResponse(BaseModel):
    text: str
    confidence: float


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/image-to-text",
    response_model=OcrResponse,
    summary="Extract text from a base64-encoded image",
)
async def image_to_text(
    body: OcrRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher")),
    ],
) -> OcrResponse:
    mime = body.mimeType.lower()
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported MIME type: {body.mimeType}",
        )

    messages = [
        {"role": "system", "content": OCR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{body.imageBase64}"
                    },
                },
                {
                    "type": "text",
                    "text": "Extract all text from this image.",
                },
            ],
        },
    ]

    try:
        raw = await call_openrouter(
            model=OCR_MODEL,
            temperature=OCR_TEMPERATURE,
            messages=messages,
        )
    except Exception as exc:
        logger.error("[ocr/image-to-text] OpenRouter error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OCR service error",
        ) from exc

    stripped = strip_json_fences(raw)
    text = raw
    confidence = 0.5

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed.get("text"), str):
            text = parsed["text"]
        if isinstance(parsed.get("confidence"), (int, float)):
            confidence = max(0.0, min(1.0, float(parsed["confidence"])))
    except (json.JSONDecodeError, AttributeError):
        logger.warning("[ocr/image-to-text] JSON parse failed; using raw content")

    return OcrResponse(text=text, confidence=confidence)
