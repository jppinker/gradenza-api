"""
Storage + persistence for Class PDF note PDF exports.
"""

from __future__ import annotations

import asyncio
from typing import Any

from gradenza_api.services.pdf_playwright import render_html_to_pdf_bytes
from gradenza_api.services.supabase_client import run_sync


EXPORTS_BUCKET = "class-pdf-note-exports"
EXPORT_TYPE_TEACHER_PDF = "pdf_teacher"


async def export_note_pdf(
    *,
    svc: Any,
    note_id: str,
    html: str,
    file_name: str,
) -> dict[str, Any]:
    """
    Render note HTML → PDF, upload to storage, and upsert export metadata.

    Returns: {pdf_url, file_name, storage_path, warnings?}
    """
    pdf_bytes, warnings = await render_html_to_pdf_bytes(html)
    storage_path = f"{note_id}/{file_name}"

    def _upload() -> None:
        svc.storage.from_(EXPORTS_BUCKET).upload(
            path=storage_path,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )

    def _upsert_export_row() -> None:
        # Ensure there's a single latest export row per note/type.
        svc.table("class_pdf_note_exports").upsert(
            {
                "note_id": note_id,
                "export_type": EXPORT_TYPE_TEACHER_PDF,
                "file_path": storage_path,
                "storage_bucket": EXPORTS_BUCKET,
                "file_name": file_name,
            },
            on_conflict="note_id,export_type",
        ).execute()

    def _sign_url() -> str:
        signed = svc.storage.from_(EXPORTS_BUCKET).create_signed_url(
            path=storage_path,
            expires_in=3600,
        )
        return signed.get("signedURL") or signed.get("signed_url") or ""

    await run_sync(_upload)
    await run_sync(_upsert_export_row)
    pdf_url = await asyncio.to_thread(_sign_url)
    result: dict[str, Any] = {
        "pdf_url": pdf_url,
        "file_name": file_name,
        "storage_path": storage_path,
    }
    if warnings:
        result["warnings"] = warnings
    return result


async def get_latest_note_pdf_url(
    *,
    svc: Any,
    note_id: str,
) -> dict[str, str] | None:
    """
    If an export exists, return {pdf_url, file_name, storage_path}, else None.
    """

    def _fetch() -> dict | None:
        result = (
            svc.table("class_pdf_note_exports")
            .select("file_path,file_name,storage_bucket")
            .eq("note_id", note_id)
            .eq("export_type", EXPORT_TYPE_TEACHER_PDF)
            .maybe_single()
            .execute()
        )
        return result.data

    row = await run_sync(_fetch)
    if not row:
        return None

    storage_path = row.get("file_path") or ""
    file_name = row.get("file_name") or ""
    bucket = row.get("storage_bucket") or EXPORTS_BUCKET

    if not storage_path:
        return None

    def _sign_url() -> str:
        signed = svc.storage.from_(bucket).create_signed_url(
            path=storage_path,
            expires_in=3600,
        )
        return signed.get("signedURL") or signed.get("signed_url") or ""

    pdf_url = await asyncio.to_thread(_sign_url)
    return {
        "pdf_url": pdf_url,
        "file_name": file_name,
        "storage_path": storage_path,
    }
