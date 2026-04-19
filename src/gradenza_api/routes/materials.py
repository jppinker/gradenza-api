"""
POST /v1/materials/generate

Generates educational material from a teacher/tutor prompt via
OpenRouter → google/gemini-3.1-flash-lite-preview.

Ported from supabase/functions/generate-material/index.ts.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import call_openrouter, strip_json_fences

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/materials", tags=["materials"])

# ── Constants ──────────────────────────────────────────────────────────────────

GENERATE_MODEL = "google/gemini-3.1-flash-lite-preview"
GENERATE_TEMPERATURE = 0.7
MAX_PROMPT_LENGTH = 2000

SYSTEM_PROMPT = """\
You are an expert educational content creator for IB, SAT, EGE, OGE, WAEC, and other \
high-school examinations.
You help teachers and tutors create high-quality reference materials, revision notes, \
practice explanations, and topic summaries.

When given a prompt, produce educational material in clear, well-structured plain text.
Use headings (prefixed with ##), bullet points, numbered lists, and LaTeX math notation \
where appropriate.

Also suggest a concise, descriptive title for the material.

Output a single JSON object with exactly two fields:
  "title": a short descriptive title (max 80 characters)
  "content": the full educational material as a plain text / markdown string

Do not include any text outside the JSON object. No markdown fences around the JSON itself."""


# ── Request / Response models ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class GenerateResponse(BaseModel):
    title: str | None
    content: str


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate educational material from a prompt",
)
async def generate_material(
    body: GenerateRequest,
    user: Annotated[
        AuthUser,
        Depends(require_roles("teacher", "tutor", "co_teacher")),
    ],
) -> GenerateResponse:
    request_id = str(uuid.uuid4())[:8]

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="prompt is required",
        )
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Prompt too long (max {MAX_PROMPT_LENGTH} characters)",
        )

    logger.info(
        "[generate-material][%s] user=%s prompt_len=%d",
        request_id,
        user.id,
        len(prompt),
    )

    try:
        raw = await call_openrouter(
            model=GENERATE_MODEL,
            temperature=GENERATE_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        logger.error("[generate-material][%s] OpenRouter error: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Generation service error",
        ) from exc

    # Parse JSON from model output
    stripped = strip_json_fences(raw)
    content = raw
    title: str | None = None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed.get("content"), str):
            content = parsed["content"]
        if isinstance(parsed.get("title"), str):
            title = parsed["title"]
    except (json.JSONDecodeError, AttributeError):
        logger.warning(
            "[generate-material][%s] JSON parse failed; using raw content", request_id
        )

    logger.info(
        "[generate-material][%s] done content_len=%d has_title=%s",
        request_id,
        len(content),
        title is not None,
    )
    return GenerateResponse(title=title, content=content)
