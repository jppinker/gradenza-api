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
from typing import Annotated

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from gradenza_api.auth import AuthUser, get_auth_user
from gradenza_api.services.supabase_client import get_service_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/submissions", tags=["submissions"])


# ── Request / Response models ──────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    force: bool = False


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
    - teacher/tutor/co_teacher/school_admin: must own the class the assignment belongs to
    """
    if user.is_internal:
        return

    def _check() -> str | None:
        svc = get_service_client()
        result = (
            svc.table("submissions")
            .select("student_id, assignments(class_id, classes(teacher_id))")
            .eq("id", submission_id)
            .maybe_single()
            .execute()
        )
        if not result.data:
            return "not_found"

        row = result.data
        if user.role == "student":
            if row.get("student_id") != user.id:
                return "forbidden"
        else:
            # teacher-side roles
            class_data = (
                row.get("assignments") or {}
            )
            if isinstance(class_data, list):
                class_data = class_data[0] if class_data else {}
            teacher_id = (
                (class_data.get("classes") or {}).get("teacher_id")
                if isinstance(class_data.get("classes"), dict)
                else None
            )
            if teacher_id != user.id:
                return "forbidden"
        return None

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

    redis: ArqRedis = request.app.state.redis
    job_id = f"process_submission_{submission_id}"

    job = await redis.enqueue_job(
        "process_submission",
        submission_id,
        body.force,
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
        row = sub_res.data or {}
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
