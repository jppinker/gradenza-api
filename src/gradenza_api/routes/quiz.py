"""
Quiz Builder routes.

GET    /v1/quiz/sessions                         list teacher's sessions
POST   /v1/quiz/sessions                         create session
GET    /v1/quiz/sessions/{session_id}            get single session (polled during generation)
DELETE /v1/quiz/sessions/{session_id}            delete session (204)
POST   /v1/quiz/sessions/{session_id}/source     save source material
POST   /v1/quiz/sessions/{session_id}/blueprint  generate question blueprint via LLM
POST   /v1/quiz/sessions/{session_id}/generate   enqueue question generation job (202)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel

from gradenza_api.auth import AuthUser, require_roles
from gradenza_api.services.openrouter import call_openrouter, strip_json_fences
from gradenza_api.services.supabase_client import get_service_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/quiz", tags=["quiz"])

_TEACHER_ROLES = ("teacher", "tutor", "co_teacher", "school_admin")
_QUIZ_MODEL = "google/gemini-2.5-flash"

VALID_MODES = frozenset({"from_source", "from_prompt", "from_questions", "personalized"})
VALID_PRESETS = frozenset({
    "quick_check", "standard_worksheet", "exam_practice",
    "support_practice", "challenge_extension", "reading_comprehension",
})


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_session(session_id: str, teacher_id: str) -> dict | None:
    svc = get_service_client()
    result = (
        svc.table("quiz_builder_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("teacher_id", teacher_id)
        .maybe_single()
        .execute()
    )
    return result.data


def _require_session(session_id: str, teacher_id: str) -> dict:
    session = _fetch_session(session_id, teacher_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


# ── GET /v1/quiz/sessions ──────────────────────────────────────────────────────

@router.get("/sessions", summary="List teacher's quiz sessions")
async def list_sessions(
    user: Annotated[AuthUser, Depends(require_roles(*_TEACHER_ROLES))],
) -> list[dict]:
    def _query() -> list[dict]:
        svc = get_service_client()
        result = (
            svc.table("quiz_builder_sessions")
            .select("*")
            .eq("teacher_id", user.id)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data or []

    return await asyncio.to_thread(_query)


# ── POST /v1/quiz/sessions ─────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    mode: str
    preset: str
    subject: str | None = None
    grade_level: str | None = None
    title: str | None = None
    class_id: str | None = None


@router.post("/sessions", status_code=status.HTTP_201_CREATED, summary="Create a quiz session")
async def create_session(
    body: CreateSessionRequest,
    user: Annotated[AuthUser, Depends(require_roles(*_TEACHER_ROLES))],
) -> dict:
    if body.mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {body.mode!r}")
    if body.preset not in VALID_PRESETS:
        raise HTTPException(status_code=400, detail=f"Invalid preset: {body.preset!r}")

    def _insert() -> dict:
        svc = get_service_client()
        result = svc.table("quiz_builder_sessions").insert({
            "teacher_id": user.id,
            "mode": body.mode,
            "preset": body.preset,
            "subject": body.subject,
            "grade_level": body.grade_level,
            "title": body.title,
            "class_id": body.class_id,
            "status": "draft",
        }).execute()
        return result.data[0] if result.data else {}

    row = await asyncio.to_thread(_insert)
    if not row:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return row


# ── GET /v1/quiz/sessions/{session_id} ────────────────────────────────────────

@router.get("/sessions/{session_id}", summary="Get a single quiz session")
async def get_session(
    session_id: str,
    user: Annotated[AuthUser, Depends(require_roles(*_TEACHER_ROLES))],
) -> dict:
    return await asyncio.to_thread(_require_session, session_id, user.id)


# ── DELETE /v1/quiz/sessions/{session_id} ─────────────────────────────────────

@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a quiz session",
)
async def delete_session(
    session_id: str,
    user: Annotated[AuthUser, Depends(require_roles(*_TEACHER_ROLES))],
) -> Response:
    await asyncio.to_thread(_require_session, session_id, user.id)

    def _delete() -> None:
        svc = get_service_client()
        svc.table("quiz_builder_sessions").delete().eq("id", session_id).execute()

    await asyncio.to_thread(_delete)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── POST /v1/quiz/sessions/{session_id}/source ────────────────────────────────

@router.post("/sessions/{session_id}/source", summary="Save source material")
async def save_source(
    session_id: str,
    source_type: Annotated[str, Form()],
    user: Annotated[AuthUser, Depends(require_roles(*_TEACHER_ROLES))],
    text: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
) -> dict:
    await asyncio.to_thread(_require_session, session_id, user.id)

    valid_types = {"pasted_text", "pdf", "txt", "docx", "prompt", "teacher_questions"}
    if source_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid source_type: {source_type!r}")

    raw_input: str = text or ""
    extracted_text: str = text or ""
    metadata: dict[str, Any] = {}

    if file is not None:
        file_bytes = await file.read()
        metadata["filename"] = file.filename
        metadata["size"] = len(file_bytes)

        if source_type == "txt":
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("latin-1", errors="replace")
            raw_input = extracted_text
        else:
            # PDF/DOCX: store filename; text extraction requires external libs
            raw_input = f"[File: {file.filename}]"
            extracted_text = ""
            metadata["note"] = "Binary file — text extraction not available server-side"

    def _upsert() -> dict:
        svc = get_service_client()
        svc.table("quiz_source_materials").delete().eq("session_id", session_id).execute()
        result = svc.table("quiz_source_materials").insert({
            "session_id": session_id,
            "source_type": source_type,
            "raw_input": raw_input,
            "extracted_text": extracted_text,
            "metadata": metadata,
        }).execute()
        return result.data[0] if result.data else {}

    row = await asyncio.to_thread(_upsert)
    return {"ok": True, "id": row.get("id")}


# ── POST /v1/quiz/sessions/{session_id}/blueprint ─────────────────────────────

_BLUEPRINT_PROMPT = """\
You are an expert curriculum designer. Generate a quiz blueprint.

Session details:
- Mode: {mode}
- Preset: {preset}
- Subject: {subject}
- Grade level: {grade_level}
- Number of questions requested: {question_count}

Source material (first 2000 chars):
{source_text}

Preset guidelines:
- quick_check: mostly easy/medium, multiple_choice and fill_blank, 1 mark each
- standard_worksheet: mix of types, easy–medium, 1–3 marks each
- exam_practice: mix of all types, medium–hard, 1–5 marks each
- support_practice: easy, short_answer and fill_blank, 1–2 marks each
- challenge_extension: medium–hard, long_answer and short_answer, 2–5 marks each
- reading_comprehension: all short_answer, medium, 2–3 marks each

Return a JSON array of exactly {question_count} objects. Each:
{{"slot":<int 1..N>,"question_type":"multiple_choice"|"fill_blank"|"short_answer"|"long_answer","difficulty":"easy"|"medium"|"hard","marks":<int 1..5>}}

Return ONLY the raw JSON array. No markdown, no explanation."""


class BlueprintRequest(BaseModel):
    question_count: int = 10


@router.post("/sessions/{session_id}/blueprint", summary="Generate question blueprint via LLM")
async def generate_blueprint(
    session_id: str,
    body: BlueprintRequest,
    user: Annotated[AuthUser, Depends(require_roles(*_TEACHER_ROLES))],
) -> dict:
    session = await asyncio.to_thread(_require_session, session_id, user.id)

    def _get_source() -> dict | None:
        svc = get_service_client()
        result = (
            svc.table("quiz_source_materials")
            .select("extracted_text")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    source = await asyncio.to_thread(_get_source)
    source_text = (source or {}).get("extracted_text") or "(no source text provided)"

    prompt = _BLUEPRINT_PROMPT.format(
        mode=session.get("mode", "from_prompt"),
        preset=session.get("preset", "standard_worksheet"),
        subject=session.get("subject") or "General",
        grade_level=session.get("grade_level") or "Secondary",
        question_count=body.question_count,
        source_text=source_text[:2000],
    )

    try:
        llm_result = await call_openrouter(
            model=_QUIZ_MODEL,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = strip_json_fences(llm_result.content)
        parsed = json.loads(raw)

        if isinstance(parsed, list):
            blueprint: list[dict] = parsed
        elif isinstance(parsed, dict):
            blueprint = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            blueprint = []
    except json.JSONDecodeError as exc:
        logger.error("[quiz/blueprint][%s] JSON parse error: %s", session_id, exc)
        raise HTTPException(status_code=502, detail="Could not parse blueprint from LLM") from exc
    except Exception as exc:
        logger.error("[quiz/blueprint][%s] LLM error: %s", session_id, exc)
        raise HTTPException(status_code=502, detail="LLM service error") from exc

    def _save_blueprint() -> None:
        svc = get_service_client()
        svc.table("quiz_blueprints").delete().eq("session_id", session_id).execute()
        svc.table("quiz_blueprints").insert({
            "session_id": session_id,
            "blueprint_json": blueprint,
        }).execute()

    await asyncio.to_thread(_save_blueprint)
    return {"blueprint_json": blueprint}


# ── POST /v1/quiz/sessions/{session_id}/generate ──────────────────────────────

@router.post(
    "/sessions/{session_id}/generate",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue quiz question generation",
)
async def start_generation(
    session_id: str,
    request: Request,
    user: Annotated[AuthUser, Depends(require_roles(*_TEACHER_ROLES))],
) -> dict:
    await asyncio.to_thread(_require_session, session_id, user.id)

    def _set_generating() -> None:
        svc = get_service_client()
        svc.table("quiz_builder_sessions").update({
            "status": "generating",
        }).eq("id", session_id).execute()

    await asyncio.to_thread(_set_generating)

    redis = request.app.state.redis
    await redis.enqueue_job(
        "generate_quiz",
        session_id=session_id,
        _job_id=f"quiz_{session_id}",
    )

    logger.info("[quiz/generate] enqueued session=%s", session_id)
    return {"enqueued": True, "session_id": session_id}
