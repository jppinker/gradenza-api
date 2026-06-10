"""
POST /v1/submissions/{submission_id}/process
  Enqueues the OCR + grading orchestrator job for a submission.
  Idempotent: ARQ deduplicates by job_id so multiple calls are safe.

GET  /v1/submissions/{submission_id}/processing-status
  Returns current status derived from DB fields.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Annotated

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from gradenza_api.auth import AuthUser, get_auth_user
from gradenza_api.services.supabase_client import get_service_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/submissions", tags=["submissions"])


# ── Request / Response models ──────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    submission_id: str
    assignment_question_id: str | None = None


class ProcessResponse(BaseModel):
    enqueued: bool
    job_id: str


class ProcessingStatusResponse(BaseModel):
    submission_id: str
    submission_status: str | None
    total_photos: int
    photos_ocr_done: int
    all_ocr_done: bool
    processing_error: str | None = None
    processing_error_at: str | None = None


# ── Auth helper: student owner OR teacher who owns the class ───────────────────

async def _authorize_submission(
    submission_id: str,
    user: AuthUser,
) -> None:
    """
    Raises 403 if the caller is not authorised to trigger processing
    for this submission.
    - internal: always allowed
    - student: must own the submission
    - teacher/tutor: must own the class OR have created the assignment
      (created_by covers classless tutor assignments that have no class)
    - co_teacher: must have an accepted class_co_teachers row for the class
    - school_admin: must share the same workspace as the class
    """
    if user.is_internal:
        return

    def _check() -> str | None:
        svc = get_service_client()
        result = (
            svc.table("submissions")
            .select("student_id, assignments(created_by, class_id, classes(teacher_id, workspace_id))")
            .eq("id", submission_id)
            .maybe_single()
            .execute()
        )
        if result is None or not result.data:
            return "not_found"

        row = result.data
        if user.role == "student":
            if row.get("student_id") != user.id:
                return "forbidden"
            return None

        # teacher-side roles
        assignment = row.get("assignments") or {}
        if isinstance(assignment, list):
            assignment = assignment[0] if assignment else {}
        class_id: str | None = assignment.get("class_id")
        created_by: str | None = assignment.get("created_by")
        classes = assignment.get("classes") or {}
        if isinstance(classes, list):
            classes = classes[0] if classes else {}
        teacher_id: str | None = classes.get("teacher_id")

        if user.role in ("teacher", "tutor"):
            # Owns the class OR created the assignment (covers classless tutor assignments).
            if teacher_id == user.id or created_by == user.id:
                return None
            return "forbidden"

        if user.role == "co_teacher":
            if not class_id:
                return "forbidden"
            co_res = (
                svc.table("class_co_teachers")
                .select("id")
                .eq("class_id", class_id)
                .eq("user_id", user.id)
                .not_.is_("accepted_at", "null")
                .maybe_single()
                .execute()
            )
            if co_res and co_res.data:
                return None
            return "forbidden"

        if user.role == "school_admin":
            if not class_id:
                return "forbidden"
            class_workspace: str | None = classes.get("workspace_id")
            if not class_workspace:
                return "forbidden"
            user_res = (
                svc.table("users")
                .select("workspace_id")
                .eq("id", user.id)
                .maybe_single()
                .execute()
            )
            user_workspace = ((user_res.data if user_res else None) or {}).get("workspace_id")
            if class_workspace == user_workspace:
                return None
            return "forbidden"

        return "forbidden"

    error = await asyncio.to_thread(_check)
    if error == "not_found":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")
    if error == "forbidden":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorised")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/{submission_id}/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue OCR + grading for a submission",
)
async def process_submission(
    submission_id: str,
    body: ProcessRequest,
    request: Request,
    user: Annotated[AuthUser, Depends(get_auth_user)],
) -> ProcessResponse:
    await _authorize_submission(submission_id, user)

    if body.submission_id != submission_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="submission_id mismatch")

    has_aqid = "assignment_question_id" in body.model_fields_set
    if has_aqid:
        def _fetch_scope() -> dict | None:
            svc = get_service_client()
            res = (
                svc.table("submissions")
                .select("id, assignment_question_id")
                .eq("id", submission_id)
                .maybe_single()
                .execute()
            )
            return res.data if res is not None else None

        scope = await asyncio.to_thread(_fetch_scope)
        if not scope:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found")

        submission_aqid = scope.get("assignment_question_id")
        # Only reject when this is a per-question submission that doesn't match the
        # requested question.  When submission_aqid is NULL (legacy full-assignment
        # submission), the caller-supplied AQID is a valid grading-scope filter and
        # must be allowed through.  Never persist processing_error for a request
        # that is rejected before the job is even enqueued.
        if submission_aqid is not None and submission_aqid != body.assignment_question_id:
            logger.warning(
                "[submissions] assignment_question_id mismatch submission=%s expected=%s got=%s",
                submission_id,
                submission_aqid,
                body.assignment_question_id,
            )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="assignment_question_id mismatch")

    redis: ArqRedis | None = request.app.state.redis
    if redis is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background queue temporarily unavailable. Please retry shortly.",
        )
    # Include unix-second timestamp so that a completed job's result (kept for
    # keep_result=3600 s) does not silently block re-enqueuing after a student
    # edits their photos.  Rapid duplicate calls within the same second are still
    # deduplicated by ARQ; edits (which take longer than 1 s to complete) always
    # produce a fresh job_id and are never dropped.
    job_id = f"process_submission_{submission_id}_{int(time.time())}"

    job = await redis.enqueue_job(
        "process_submission",
        submission_id,
        body.assignment_question_id,
        has_aqid,
        _job_id=job_id,
    )

    if job is None:
        # ARQ returns None when a job with that ID already exists in the queue
        logger.info(
            "[submissions] job already queued for submission=%s", submission_id
        )
    else:
        logger.info(
            "[submissions] enqueued job_id=%s submission=%s", job_id, submission_id
        )

    return ProcessResponse(enqueued=True, job_id=job_id)


@router.get(
    "/{submission_id}/processing-status",
    response_model=ProcessingStatusResponse,
    summary="Get the current OCR/grading status for a submission",
)
async def get_processing_status(
    submission_id: str,
    user: Annotated[AuthUser, Depends(get_auth_user)],
) -> ProcessingStatusResponse:
    await _authorize_submission(submission_id, user)

    def _fetch() -> dict:
        svc = get_service_client()

        sub_res = (
            svc.table("submissions")
            .select("status, processing_error, processing_error_at")
            .eq("id", submission_id)
            .maybe_single()
            .execute()
        )
        row = (sub_res.data if sub_res is not None else None) or {}
        submission_status = row.get("status")
        processing_error = row.get("processing_error")
        processing_error_at = row.get("processing_error_at")

        photos_res = (
            svc.table("submission_photos")
            .select("ocr_done_at")
            .eq("submission_id", submission_id)
            .execute()
        )
        photos = photos_res.data or []
        total = len(photos)
        done = sum(1 for p in photos if p.get("ocr_done_at") is not None)

        return {
            "submission_status": submission_status,
            "total_photos": total,
            "photos_ocr_done": done,
            "processing_error": processing_error,
            "processing_error_at": processing_error_at,
        }

    data = await asyncio.to_thread(_fetch)

    return ProcessingStatusResponse(
        submission_id=submission_id,
        submission_status=data["submission_status"],
        total_photos=data["total_photos"],
        photos_ocr_done=data["photos_ocr_done"],
        all_ocr_done=data["total_photos"] > 0
        and data["photos_ocr_done"] == data["total_photos"],
        processing_error=data["processing_error"],
        processing_error_at=data["processing_error_at"],
    )
