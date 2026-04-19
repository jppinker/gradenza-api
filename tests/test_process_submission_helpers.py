"""Unit tests for pure helper functions in process_submission.

No database, network, or environment variables required.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from gradenza_api.jobs.process_submission import (
    PHOTO_FETCH_ATTEMPTS,
    ParsedMarkschemeStep,
    ParsedPart,
    _clamp01,
    _classify_trust_layer,
    _detect_amber,
    _fetch_photos_retrying,
    _mime_for_path,
    _total_available_marks,
    process_submission,
)


# ── _clamp01 ──────────────────────────────────────────────────────────────────

def test_clamp01_within_range():
    assert _clamp01(0.5) == 0.5

def test_clamp01_clamps_above_one():
    assert _clamp01(1.5) == 1.0

def test_clamp01_clamps_below_zero():
    assert _clamp01(-0.1) == 0.0


# ── _mime_for_path ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("student/sub/attempt_1/001.png", "image/png"),
    ("student/sub/attempt_1/001.jpg", "image/jpeg"),
    ("student/sub/attempt_1/001.jpeg", "image/jpeg"),
    ("student/sub/attempt_1/scan.pdf", "application/pdf"),
    ("student/sub/attempt_1/page.webp", "image/webp"),
    ("student/sub/attempt_1/file.PNG", "image/jpeg"),  # case-sensitive — fallback
])
def test_mime_for_path(path, expected):
    assert _mime_for_path(path) == expected


# ── _classify_trust_layer ─────────────────────────────────────────────────────

def test_trust_layer_amber():
    assert _classify_trust_layer(amber_flag=True, marks_awarded=3, marks_available=5) == "layer2_amber"

def test_trust_layer_zero_marks():
    assert _classify_trust_layer(amber_flag=False, marks_awarded=0, marks_available=5) == "layer3_incorrect"

def test_trust_layer_auto():
    assert _classify_trust_layer(amber_flag=False, marks_awarded=4, marks_available=5) == "layer1_auto"

def test_trust_layer_zero_marks_zero_available():
    # 0/0 — no amber, but also not "incorrect" (nothing to award)
    assert _classify_trust_layer(amber_flag=False, marks_awarded=0, marks_available=0) == "layer1_auto"


# ── _detect_amber ─────────────────────────────────────────────────────────────

def test_detect_amber_crossed_out():
    amber, reason = _detect_amber("[CROSSED OUT: x + 1]", {"confidence": 0.9})
    assert amber is True
    assert reason is not None

def test_detect_amber_none_result():
    amber, reason = _detect_amber("clean text", None)
    assert amber is True
    assert "manual review" in reason.lower()

def test_detect_amber_low_confidence():
    amber, reason = _detect_amber("clean text", {"confidence": 0.5})
    assert amber is True

def test_detect_amber_clean():
    amber, reason = _detect_amber("clean text", {"confidence": 0.95})
    assert amber is False
    assert reason is None


# ── _total_available_marks ────────────────────────────────────────────────────

def test_total_marks_from_steps():
    steps = [
        ParsedMarkschemeStep(part_label="a", description="", marks=2, mark_type="M"),
        ParsedMarkschemeStep(part_label="b", description="", marks=3, mark_type="A"),
        ParsedMarkschemeStep(part_label="c", description="", marks=1, mark_type="AG"),  # AG excluded
    ]
    parts = [ParsedPart(label="a", marks=99, text=None)]
    assert _total_available_marks(steps, parts) == 5  # 2+3, AG skipped

def test_total_marks_fallback_to_parts():
    steps: list[ParsedMarkschemeStep] = []
    parts = [
        ParsedPart(label="a", marks=3, text=None),
        ParsedPart(label="b", marks=2, text=None),
    ]
    assert _total_available_marks(steps, parts) == 5


# ── _fetch_photos_retrying ────────────────────────────────────────────────────

async def test_fetch_photos_retrying_returns_on_first_non_empty():
    calls = []

    def fetch():
        calls.append(1)
        return [{"id": "photo-1"}]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await _fetch_photos_retrying(fetch, "sub-1", max_attempts=3, retry_delay=0)

    assert result == [{"id": "photo-1"}]
    assert len(calls) == 1
    mock_sleep.assert_not_called()


async def test_fetch_photos_retrying_retries_on_empty_then_succeeds():
    calls = []

    def fetch():
        calls.append(1)
        return [] if len(calls) < 2 else [{"id": "photo-1"}]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await _fetch_photos_retrying(fetch, "sub-1", max_attempts=3, retry_delay=0)

    assert result == [{"id": "photo-1"}]
    assert len(calls) == 2
    assert mock_sleep.call_count == 1


async def test_fetch_photos_retrying_exhausts_all_attempts():
    calls = []

    def fetch():
        calls.append(1)
        return []

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await _fetch_photos_retrying(fetch, "sub-1", max_attempts=3, retry_delay=0)

    assert result == []
    assert len(calls) == PHOTO_FETCH_ATTEMPTS
    assert mock_sleep.call_count == PHOTO_FETCH_ATTEMPTS - 1  # sleep between attempts, not after last


# ── Integration: process_submission failure paths ─────────────────────────────
#
# These tests use per-table mocks (side_effect on svc.table) so we can
# assert exactly which update() calls land on submissions vs other tables.
#
# submissions.update() is called TWICE on every failure path:
#   1. _clear_processing_error() at job start → {processing_error: None, ...}
#   2. _set_processing_error(code)            → {processing_error: "<code>", ...}

def _make_svc(photos_data: list[dict]) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_svc, mock_photos_tbl, mock_subs_tbl) with photos_data wired up."""
    mock_photos_tbl = MagicMock()
    (
        mock_photos_tbl.select.return_value
        .eq.return_value
        .order.return_value
        .execute.return_value
        .data
    ) = photos_data

    mock_subs_tbl = MagicMock()

    mock_svc = MagicMock()
    mock_svc.table.side_effect = (
        lambda name: mock_photos_tbl if name == "submission_photos" else mock_subs_tbl
    )
    return mock_svc, mock_photos_tbl, mock_subs_tbl


def _assert_error_writes(mock_subs_tbl: MagicMock, expected_code: str) -> None:
    """Assert the two expected submissions.update() calls: clear then set."""
    calls = mock_subs_tbl.update.call_args_list
    assert len(calls) == 2, f"expected 2 update calls, got {len(calls)}: {calls}"
    # First call clears any previous error
    assert calls[0].args[0] == {"processing_error": None, "processing_error_at": None}
    # Second call records the new error
    payload = calls[1].args[0]
    assert payload["processing_error"] == expected_code
    assert payload["processing_error_at"] is not None


async def test_process_submission_no_photos_writes_error_and_exits():
    """No photos after all retries → processing_error='no_photos', no OCR/grading."""
    mock_svc, _, mock_subs_tbl = _make_svc(photos_data=[])

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=mock_svc),
        patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        patch("gradenza_api.jobs.process_submission._run_ocr") as mock_ocr,
        patch("gradenza_api.jobs.process_submission.call_openrouter") as mock_llm,
    ):
        result = await process_submission({}, submission_id="sub-dead", force=False)

    assert result == {"submission_id": "sub-dead", "error": "no_photos"}
    assert mock_sleep.call_count == PHOTO_FETCH_ATTEMPTS - 1
    _assert_error_writes(mock_subs_tbl, "no_photos")
    mock_ocr.assert_not_called()
    mock_llm.assert_not_called()
    # grading_results.upsert() never reached
    mock_subs_tbl.upsert.assert_not_called()
    assert all(c.args[0] != "grading_results" for c in mock_svc.table.call_args_list)


async def test_process_submission_storage_download_error_writes_error_and_exits():
    """Storage download returns None → processing_error='storage_download_error'.

    No OCR or grading steps run after the failed download.
    """
    photo_row = {
        "id": "ph-1",
        "submission_id": "sub-x",
        "page_number": 1,
        "storage_path": "student/sub-x/attempt_1/001.png",
        "ocr_done_at": None,
    }
    mock_svc, _, mock_subs_tbl = _make_svc(photos_data=[photo_row])
    # Storage download returns None without raising (simulates unreachable object)
    mock_svc.storage.from_.return_value.download.return_value = None

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=mock_svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("gradenza_api.jobs.process_submission._run_ocr") as mock_ocr,
        patch("gradenza_api.jobs.process_submission.call_openrouter") as mock_llm,
    ):
        result = await process_submission({}, submission_id="sub-x", force=False)

    assert result == {"submission_id": "sub-x", "error": "storage_download_error"}
    _assert_error_writes(mock_subs_tbl, "storage_download_error")
    mock_ocr.assert_not_called()
    mock_llm.assert_not_called()
    mock_subs_tbl.upsert.assert_not_called()
    assert all(c.args[0] != "grading_results" for c in mock_svc.table.call_args_list)


# ── Integration: grading_results upsert ──────────────────────────────────────
#
# These tests drive process_submission all the way through grading by using
# pre-OCR'd photos (ocr_done_at set) so the OCR loop is skipped.
#
# submissions.update() is called 3 times on the success path:
#   1. _clear_processing_error()  → {processing_error: None, ...}
#   2. _advance_to_ocr_done()     → {status: "ocr_done"}
#   3. _advance_to_graded()       → {status: "graded", graded_at: ...}
#
# On failure (upsert raises):
#   1. _clear_processing_error()
#   2. _advance_to_ocr_done()
#   3. _set_processing_error("db_write_error")

_PHOTO_DONE = {
    "id": "ph-1",
    "submission_id": "sub-grade",
    "page_number": 1,
    "storage_path": "s/p.jpg",
    "ocr_done_at": "2024-01-01T00:00:00+00:00",
}

_SUBMISSION_ROW = {
    "id": "sub-grade",
    "status": "ocr_done",
    "student_id": "stu-1",
    "assignment_id": "assign-1",
    "assignments": {
        "id": "assign-1",
        "grading_style": "ib_style",
        "class_id": "cls-1",
        "classes": {
            "id": "cls-1",
            "exam_system_id": "es-1",
            "exam_systems": {"id": "es-1", "code": "IB"},
        },
    },
}

_PROMPT_ROW = {"id": "pv-1", "prompt_text": "Grade IB style.", "version_tag": "v1"}

_AQ_ROW = {
    "id": "aq-1",
    "position": 1,
    "question_id": "q-1",
    "questions": {
        "id": "q-1",
        "source_id": "123",
        "source_table": "ib_math_questionbank",
        "diagram_required": False,
        "ft_eligible_parts": [],
        "ft_dependencies": {},
    },
}

_IB_ROW = {
    "id": 123,
    "domain": "Algebra",
    "theme": "Sequences",
    "theme_slug": "seq",
    "level": "SL",
    "difficulty": "medium",
    "problem_text": "Solve for x.",
    "question_media_count": 0,
    "parts_count": 1,
    "parts_json": [{"label": "a", "marks": 5, "text": "Find x."}],
    "markscheme_steps_count": 1,
    "markscheme_steps_json": [{"description": "x=2", "marks": 5, "mark_type": "M"}],
}

_LLM_RESPONSE = json.dumps({
    "total_marks_awarded": 3,
    "total_method_marks": 2,
    "total_accuracy_marks": 1,
    "total_ft_marks": 0,
    "confidence": 0.85,
    "feedback_text": "Good attempt.",
    "overall_amber_flag": False,
    "overall_amber_reason": None,
    "full_assessment": "Student found x=2.",
    "parts": [{"part_label": "a", "extracted_answer": "x=2", "ft_applied": False}],
})


def _make_grading_svc(upsert_raises: bool = False):
    """Build a multi-table mock svc wired for the full grading happy path."""
    photos_tbl = MagicMock()
    subs_tbl = MagicMock()
    prompt_tbl = MagicMock()
    aq_tbl = MagicMock()
    ib_tbl = MagicMock()
    gr_tbl = MagicMock()

    def photos_select(fields):
        m = MagicMock()
        if fields.strip() == "ocr_done_at":
            # _check_all_done: no .order() call
            m.eq.return_value.execute.return_value.data = [_PHOTO_DONE]
        elif "ocr_text" in fields:
            # _fetch_ocr_pages
            m.eq.return_value.order.return_value.execute.return_value.data = [
                {"page_number": 1, "ocr_text": "Student answer: x=2", "ocr_confidence": 0.9}
            ]
        else:
            # _fetch_photos
            m.eq.return_value.order.return_value.execute.return_value.data = [_PHOTO_DONE]
        return m

    photos_tbl.select.side_effect = photos_select

    # submissions: select for _fetch_submission
    subs_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = _SUBMISSION_ROW

    # grading_prompt_versions
    prompt_tbl.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = _PROMPT_ROW

    # assignment_questions
    aq_tbl.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [_AQ_ROW]

    # ib_math_questionbank
    ib_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = _IB_ROW

    # grading_results upsert — succeed or raise based on flag
    if upsert_raises:
        gr_tbl.upsert.return_value.execute.side_effect = Exception("db error")

    table_map = {
        "submission_photos": photos_tbl,
        "submissions": subs_tbl,
        "grading_prompt_versions": prompt_tbl,
        "assignment_questions": aq_tbl,
        "ib_math_questionbank": ib_tbl,
        "grading_results": gr_tbl,
    }
    svc = MagicMock()
    svc.table.side_effect = lambda name: table_map.get(name, MagicMock())

    return svc, subs_tbl, gr_tbl


async def test_grading_results_upsert_called_with_keyword_on_conflict():
    """upsert() must use on_conflict as a keyword arg (not a positional dict)."""
    svc, _, gr_tbl = _make_grading_svc()

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_LLM_RESPONSE,
        ),
    ):
        await process_submission({}, submission_id="sub-grade", force=False)

    gr_tbl.upsert.assert_called_once()
    pos_args = gr_tbl.upsert.call_args.args
    kwargs = gr_tbl.upsert.call_args.kwargs
    assert len(pos_args) == 1, f"upsert() should receive exactly 1 positional arg, got {len(pos_args)}"
    assert kwargs.get("on_conflict") == "submission_id,assignment_question_id"
    payload = pos_args[0]
    assert payload["submission_id"] == "sub-grade"
    assert payload["assignment_question_id"] == "aq-1"


async def test_grading_results_upsert_success_clears_error_and_advances_to_graded():
    """Successful upsert: processing_error stays cleared, status advances to 'graded'."""
    svc, subs_tbl, gr_tbl = _make_grading_svc()

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_LLM_RESPONSE,
        ),
    ):
        result = await process_submission({}, submission_id="sub-grade", force=False)

    assert result["questions_graded"] == 1
    assert result["errors"] == 0

    update_calls = subs_tbl.update.call_args_list
    assert len(update_calls) == 3, f"expected 3 submissions.update calls, got {len(update_calls)}"
    assert update_calls[0].args[0] == {"processing_error": None, "processing_error_at": None}
    assert update_calls[1].args[0] == {"status": "ocr_done"}
    graded_payload = update_calls[2].args[0]
    assert graded_payload["status"] == "graded"
    assert "processing_error" not in graded_payload


async def test_grading_results_upsert_failure_sets_db_write_error():
    """When upsert raises, processing_error='db_write_error' and status stays 'ocr_done'."""
    svc, subs_tbl, gr_tbl = _make_grading_svc(upsert_raises=True)

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_LLM_RESPONSE,
        ),
    ):
        result = await process_submission({}, submission_id="sub-grade", force=False)

    assert result["errors"] == 1
    assert result["questions_graded"] == 0

    update_calls = subs_tbl.update.call_args_list
    assert len(update_calls) == 3, f"expected 3 submissions.update calls, got {len(update_calls)}"
    assert update_calls[0].args[0] == {"processing_error": None, "processing_error_at": None}
    assert update_calls[1].args[0] == {"status": "ocr_done"}
    error_payload = update_calls[2].args[0]
    assert error_payload["processing_error"] == "db_write_error"
    assert error_payload["processing_error_at"] is not None
    assert not any(c.args[0].get("status") == "graded" for c in update_calls)
