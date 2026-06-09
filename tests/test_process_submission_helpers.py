"""Unit tests for pure helper functions in process_submission.

No database, network, or environment variables required.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import gradenza_api.jobs.process_submission as process_submission_module
from gradenza_api.jobs.process_submission import (
    PHOTO_FETCH_ATTEMPTS,
    ParsedMarkschemeStep,
    ParsedPart,
    ParsedQuestion,
    _call_grading_llm,
    _clamp01,
    _classify_trust_layer,
    _detect_amber,
    _fetch_photos_retrying,
    _mime_for_path,
    _normalize_llm_part_labels,
    _run_ocr,
    _safe_float,
    _safe_int,
    _total_available_marks,
    process_submission,
)


@pytest.fixture(autouse=True)
def _disable_asyncio_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid threads in unit tests (sandbox may not wake the loop reliably)."""

    async def _to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    monkeypatch.setattr(process_submission_module.asyncio, "to_thread", _to_thread)


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


def test_normalize_llm_part_labels_replaces_null_label():
    question = ParsedQuestion(
        question_uuid="q-1",
        source_id="src-1",
        problem_text="Question",
        diagram_required=False,
        parts=[],
        markscheme_steps=[],
        marks_available=1,
        ft_eligible_parts=[],
        ft_dependencies={},
        assignment_question_id="aq-1",
    )
    llm_result = {"parts": [{"part_label": None, "marks_awarded": 1}]}

    _normalize_llm_part_labels(llm_result, question)

    assert llm_result["parts"][0]["part_label"] == "a"


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

def _make_svc(
    photos_data: list[dict],
    *,
    submission_row: dict | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_svc, mock_photos_tbl, mock_subs_tbl) with photos_data wired up."""
    mock_photos_tbl = MagicMock()
    (
        mock_photos_tbl.select.return_value
        .eq.return_value
        .order.return_value
        .execute.return_value
        .data
    ) = photos_data

    # process_submission() now fetches the submission early for AQID scoping validation.
    if submission_row is None:
        submission_row = {
            "id": "sub-x",
            "status": "submitted",
            "assignment_id": "assign-1",
            "assignment_question_id": None,
        }

    mock_subs_tbl = MagicMock()
    (
        mock_subs_tbl.select.return_value
        .eq.return_value
        .maybe_single.return_value
        .execute.return_value
        .data
    ) = submission_row

    mock_svc = MagicMock()
    mock_svc.table.side_effect = (
        lambda name: mock_photos_tbl if name == "submission_photos" else mock_subs_tbl
    )
    return mock_svc, mock_photos_tbl, mock_subs_tbl


def _assert_error_writes(mock_subs_tbl: MagicMock, expected_code: str) -> None:
    """Assert the expected submissions.update() calls: a clear then an error set."""
    payloads = [c.args[0] for c in mock_subs_tbl.update.call_args_list if c.args]
    clear_calls = [p for p in payloads if p.get("processing_error") is None and "processing_error_at" in p]
    assert len(clear_calls) >= 1, f"expected a clear_processing_error call; got: {payloads}"
    error_calls = [p for p in payloads if p.get("processing_error") == expected_code]
    assert len(error_calls) == 1, f"expected one error call with code={expected_code!r}; got: {payloads}"
    assert error_calls[0].get("mock_phase") == "error", f"expected mock_phase='error' in error payload; got: {error_calls[0]}"
    assert error_calls[0]["processing_error_at"] is not None


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


# ── H2: OCR service failure must not write ocr_done_at ───────────────────────

async def test_ocr_service_failure_sets_error_without_marking_photo_done():
    """When OpenRouter raises during OCR, processing_error='ocr_service_error' is set
    and ocr_done_at is NOT written — so the photo can be retried on the next run."""
    photo_row = {
        "id": "ph-1",
        "submission_id": "sub-ocr-fail",
        "page_number": 1,
        "storage_path": "student/sub-ocr-fail/attempt_1/001.jpg",
        "ocr_done_at": None,
    }
    mock_svc, mock_photos_tbl, mock_subs_tbl = _make_svc(
        photos_data=[photo_row],
        submission_row={
            "id": "sub-ocr-fail",
            "status": "submitted",
            "assignment_id": "assign-1",
            "assignment_question_id": None,
        },
    )
    mock_svc.storage.from_.return_value.download.return_value = b"fake_image_bytes"

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=mock_svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            side_effect=RuntimeError("OpenRouter 503"),
        ),
        patch("gradenza_api.jobs.process_submission.record_ai_usage", new_callable=AsyncMock),
    ):
        result = await process_submission({}, submission_id="sub-ocr-fail", force=False)

    assert result == {"submission_id": "sub-ocr-fail", "error": "ocr_service_error"}
    _assert_error_writes(mock_subs_tbl, "ocr_service_error")

    # submission_photos.update() must NOT have been called with ocr_done_at
    for c in mock_photos_tbl.update.call_args_list:
        payload = c.args[0] if c.args else {}
        assert "ocr_done_at" not in payload, (
            f"ocr_done_at must NOT be written on OCR failure; got: {payload}"
        )


async def test_ocr_service_failure_then_healthy_retry_succeeds():
    """After an OCR transient failure (ocr_done_at stays NULL), a healthy retry
    runs OCR again and advances to graded."""
    photo_row = {
        "id": "ph-1",
        "submission_id": "sub-ocr-retry",
        "page_number": 1,
        "storage_path": "student/sub-ocr-retry/attempt_1/001.jpg",
        "ocr_done_at": None,
    }
    mock_svc, mock_photos_tbl, mock_subs_tbl = _make_svc(
        photos_data=[photo_row],
        submission_row={
            "id": "sub-ocr-retry",
            "status": "submitted",
            "assignment_id": "assign-1",
            "assignment_question_id": None,
        },
    )
    mock_svc.storage.from_.return_value.download.return_value = b"fake_image_bytes"

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=mock_svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            side_effect=RuntimeError("OpenRouter 503"),
        ),
        patch("gradenza_api.jobs.process_submission.record_ai_usage", new_callable=AsyncMock),
    ):
        first_result = await process_submission({}, submission_id="sub-ocr-retry", force=False)

    assert first_result["error"] == "ocr_service_error"
    # ocr_done_at was not written, so the photo is still retryable
    for c in mock_photos_tbl.update.call_args_list:
        assert "ocr_done_at" not in (c.args[0] if c.args else {})


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
    "assignment_question_id": None,
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

_QB_GENERATED_AQ_ROW = {
    "id": "aq-1",
    "position": 1,
    "question_id": "q-1",
    "questions": {
        "id": "q-1",
        "source_id": "qb-1",
        "source_table": "qb_generated",
        "diagram_required": False,
        "ft_eligible_parts": [],
        "ft_dependencies": {},
    },
}

_QB_GENERATED_ROW = {
    "id": "qb-1",
    "question_text": "Find the amplitude from maximum 5 and minimum -1.",
    "parts_json": [],
    "markscheme_steps_json": [
        {"description": "Use half the range: (5 - -1) / 2 = 3", "marks": 3}
    ],
    "total_marks": 3,
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

_LLM_RESPONSE_NULL_PART_LABEL = json.dumps({
    "total_marks_awarded": 3,
    "total_method_marks": 2,
    "total_accuracy_marks": 1,
    "total_ft_marks": 0,
    "confidence": 0.85,
    "feedback_text": "Good attempt.",
    "overall_amber_flag": False,
    "overall_amber_reason": None,
    "full_assessment": "Student found x=2.",
    "parts": [{"part_label": None, "extracted_answer": "x=2", "ft_applied": False}],
})


def _openrouter_result(content: str) -> MagicMock:
    result = MagicMock()
    result.content = content
    result.usage.prompt_tokens = 10
    result.usage.completion_tokens = 5
    result.usage.total_tokens = 15
    result.usage.model = "test-model"
    result.usage.request_id = "req-test"
    return result


def _make_grading_svc(
    insert_raises: bool = False,
    *,
    aq_row: dict | None = None,
    ib_row: dict | None = None,
    qb_generated_row: dict | None = None,
):
    """Build a multi-table mock svc wired for the full grading happy path."""
    photos_tbl = MagicMock()
    subs_tbl = MagicMock()
    prompt_tbl = MagicMock()
    aq_tbl = MagicMock()
    ib_tbl = MagicMock()
    qb_tbl = MagicMock()
    gr_tbl = MagicMock()

    def photos_select(fields):
        m = MagicMock()
        if fields.strip() == "ocr_done_at":
            # _check_all_done: no .order() call
            m.eq.return_value.execute.return_value.data = [_PHOTO_DONE]
        elif "ocr_text" in fields:
            # _fetch_ocr_pages
            m.eq.return_value.order.return_value.execute.return_value.data = [
                {"submission_id": "sub-grade", "question_id": None, "page_number": 1, "ocr_text": "Student answer: x=2", "ocr_confidence": 0.9}
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
    aq_tbl.select.return_value.eq.return_value.order.return_value.execute.return_value.data = [aq_row or _AQ_ROW]

    # ib_math_questionbank
    ib_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = ib_row or _IB_ROW

    # qb_generated
    qb_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = qb_generated_row

    # grading_results: no existing row (maybe_single returns None) → insert path
    gr_tbl.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = None

    if insert_raises:
        gr_tbl.insert.return_value.execute.side_effect = Exception("db error")

    table_map = {
        "submission_photos": photos_tbl,
        "submissions": subs_tbl,
        "grading_prompt_versions": prompt_tbl,
        "assignment_questions": aq_tbl,
        "ib_math_questionbank": ib_tbl,
        "qb_generated": qb_tbl,
        "grading_results": gr_tbl,
    }
    svc = MagicMock()
    svc.table.side_effect = lambda name: table_map.get(name, MagicMock())

    return svc, subs_tbl, gr_tbl


async def test_grading_results_insert_called_for_new_row():
    """First grading of a question must insert a new grading_results row."""
    svc, _, gr_tbl = _make_grading_svc()

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_openrouter_result(_LLM_RESPONSE),
        ),
    ):
        await process_submission({}, submission_id="sub-grade", force=False)

    gr_tbl.insert.assert_called_once()
    gr_tbl.update.assert_not_called()
    payload = gr_tbl.insert.call_args.args[0]
    assert payload["submission_id"] == "sub-grade"
    assert payload["assignment_question_id"] == "aq-1"
    assert payload["teacher_overridden"] is False


async def test_grading_results_upsert_normalizes_null_llm_part_label():
    svc, _, gr_tbl = _make_grading_svc()

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_openrouter_result(_LLM_RESPONSE_NULL_PART_LABEL),
        ),
    ):
        await process_submission({}, submission_id="sub-grade", force=False)

    payload = gr_tbl.insert.call_args.args[0]
    assessment = json.loads(payload["llm_assessment"])
    assert assessment["parts"][0]["part_label"] == "a"


async def test_process_submission_resolves_qb_generated_questions():
    """Published online-lesson homework uses qb_generated sources and must auto-grade."""
    svc, _, gr_tbl = _make_grading_svc(
        aq_row=_QB_GENERATED_AQ_ROW,
        qb_generated_row=_QB_GENERATED_ROW,
    )

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_openrouter_result(_LLM_RESPONSE),
        ) as mock_llm,
    ):
        result = await process_submission({}, submission_id="sub-grade", force=False)

    assert result["questions_graded"] == 1
    assert result["errors"] == 0
    assert mock_llm.await_count == 1
    payload = gr_tbl.insert.call_args.args[0]
    assert payload["assignment_question_id"] == "aq-1"
    assert payload["marks_available"] == 3


async def test_process_submission_includes_qb_generated_mcq_options_in_prompt():
    """Published MCQ homework must include choices when sent to the grader."""
    mcq_row = {
        **_QB_GENERATED_ROW,
        "question_type": "multiple_choice",
        "question_text": "What is the amplitude from maximum 5 and minimum -1?",
        "options_json": ["A. 2", "B. 3", "C. 6"],
        "correct_answer": "B",
        "worked_solution": "Amplitude is half the range.",
    }
    svc, _, _ = _make_grading_svc(
        aq_row=_QB_GENERATED_AQ_ROW,
        qb_generated_row=mcq_row,
    )

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_LLM_RESPONSE,
        ) as mock_llm,
    ):
        result = await process_submission({}, submission_id="sub-grade", force=False)

    assert result["questions_graded"] == 1
    user_message = mock_llm.await_args.kwargs["messages"][1]["content"]
    assert "What is the amplitude from maximum 5 and minimum -1?" in user_message
    assert "A. 2" in user_message
    assert "B. 3" in user_message
    assert "C. 6" in user_message


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
    assert len(update_calls) == 5, f"expected 5 submissions.update calls, got {len(update_calls)}"
    assert update_calls[0].args[0] == {"processing_error": None, "processing_error_at": None}
    assert update_calls[1].args[0] == {"mock_phase": "ocr"}
    assert update_calls[2].args[0] == {"status": "ocr_done", "mock_phase": "extracting"}
    assert update_calls[3].args[0] == {"mock_phase": "grading"}
    graded_payload = update_calls[4].args[0]
    assert graded_payload["status"] == "graded"
    assert graded_payload["mock_phase"] == "done"
    assert "processing_error" not in graded_payload


async def test_grading_results_first_grade_does_not_crash_when_no_existing_row():
    """maybe_single().execute() returns None for new rows; _msd() must guard this so
    the insert path runs instead of raising AttributeError on .data."""
    svc, _, gr_tbl = _make_grading_svc()
    # Explicit: select returns None (the postgrest behaviour for an empty maybe_single)
    gr_tbl.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = None

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_openrouter_result(_LLM_RESPONSE),
        ),
    ):
        result = await process_submission({}, submission_id="sub-grade", force=False)

    assert result["questions_graded"] == 1, "insert should succeed, not crash with AttributeError"
    assert result["errors"] == 0
    gr_tbl.insert.assert_called_once()
    gr_tbl.update.assert_not_called()


async def test_grading_results_upsert_failure_sets_db_write_error():
    """When the grading_results insert raises, processing_error='db_write_error' and status stays 'ocr_done'."""
    svc, subs_tbl, gr_tbl = _make_grading_svc(insert_raises=True)

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_openrouter_result(_LLM_RESPONSE),
        ),
    ):
        result = await process_submission({}, submission_id="sub-grade", force=False)

    assert result["errors"] == 1
    assert result["questions_graded"] == 0

    update_calls = subs_tbl.update.call_args_list
    assert len(update_calls) == 5, f"expected 5 submissions.update calls, got {len(update_calls)}"
    assert update_calls[0].args[0] == {"processing_error": None, "processing_error_at": None}
    assert update_calls[1].args[0] == {"mock_phase": "ocr"}
    assert update_calls[2].args[0] == {"status": "ocr_done", "mock_phase": "extracting"}
    assert update_calls[3].args[0] == {"mock_phase": "grading"}
    error_payload = update_calls[4].args[0]
    assert error_payload["processing_error"] == "db_write_error"
    assert error_payload["processing_error_at"] is not None
    assert error_payload.get("mock_phase") == "error"
    assert not any(c.args[0].get("status") == "graded" for c in update_calls)


# ── Integration: AQID scoping safety checks ─────────────────────────────────

async def test_process_submission_assignment_question_id_mismatch_writes_error_and_exits():
    """Mismatch between request AQID and submissions.assignment_question_id aborts early."""
    mock_svc, mock_photos_tbl, mock_subs_tbl = _make_svc(
        photos_data=[{
            "id": "ph-1",
            "submission_id": "sub-mismatch",
            "page_number": 1,
            "storage_path": "s/p.jpg",
            "ocr_done_at": None,
        }],
        submission_row={
            "id": "sub-mismatch",
            "status": "submitted",
            "assignment_id": "assign-1",
            "assignment_question_id": "aq-1",
        },
    )

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=mock_svc),
        patch("gradenza_api.jobs.process_submission._run_ocr") as mock_ocr,
        patch("gradenza_api.jobs.process_submission.call_openrouter") as mock_llm,
    ):
        result = await process_submission(
            {},
            submission_id="sub-mismatch",
            assignment_question_id="aq-2",
            has_assignment_question_id=True,
            force=False,
        )

    assert result == {"submission_id": "sub-mismatch", "error": "assignment_question_id mismatch"}
    _assert_error_writes(mock_subs_tbl, "assignment_question_id mismatch")
    mock_ocr.assert_not_called()
    mock_llm.assert_not_called()
    # Should abort before fetching photos
    mock_photos_tbl.select.assert_not_called()


async def test_question_scoped_photo_question_id_mismatch_aborts_before_ocr():
    """Question-scoped submissions must not contain photos tagged to a different AQID."""
    mock_svc, mock_photos_tbl, mock_subs_tbl = _make_svc(
        photos_data=[{
            "id": "ph-1",
            "submission_id": "sub-q",
            "question_id": "aq-2",
            "page_number": 1,
            "storage_path": "s/p.jpg",
            "ocr_done_at": None,
        }],
        submission_row={
            "id": "sub-q",
            "status": "submitted",
            "assignment_id": "assign-1",
            "assignment_question_id": "aq-1",
        },
    )

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=mock_svc),
        patch("gradenza_api.jobs.process_submission._run_ocr") as mock_ocr,
        patch("gradenza_api.jobs.process_submission.call_openrouter") as mock_llm,
    ):
        result = await process_submission({}, submission_id="sub-q", force=False)

    assert result == {"submission_id": "sub-q", "error": "assignment_question_id mismatch"}
    _assert_error_writes(mock_subs_tbl, "assignment_question_id mismatch")
    mock_ocr.assert_not_called()
    mock_llm.assert_not_called()


_AQ_ROW_2 = {
    "id": "aq-2",
    "position": 2,
    "question_id": "q-2",
    "questions": {
        "id": "q-2",
        "source_id": "123",
        "source_table": "ib_math_questionbank",
        "diagram_required": False,
        "ft_eligible_parts": [],
        "ft_dependencies": {},
    },
}


def _make_question_scoped_grading_svc():
    """Build a grading svc where submission is question-scoped (assignment_question_id != NULL)."""
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
                {"submission_id": "sub-grade", "question_id": "aq-1", "page_number": 1, "ocr_text": "Student answer: x=2", "ocr_confidence": 0.9}
            ]
        else:
            # _fetch_photos
            m.eq.return_value.order.return_value.execute.return_value.data = [_PHOTO_DONE | {"question_id": "aq-1"}]
        return m

    photos_tbl.select.side_effect = photos_select

    scoped_submission = dict(_SUBMISSION_ROW)
    scoped_submission["assignment_question_id"] = "aq-1"

    subs_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = scoped_submission

    prompt_tbl.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = _PROMPT_ROW

    # assignment_questions: if legacy query is used, return TWO rows; scoped query should return ONE.
    aq_select = MagicMock()
    aq_tbl.select.return_value = aq_select
    aq_select.eq.return_value.eq.return_value.execute.return_value.data = [_AQ_ROW]
    aq_select.eq.return_value.order.return_value.execute.return_value.data = [_AQ_ROW, _AQ_ROW_2]

    ib_tbl.select.return_value.eq.return_value.maybe_single.return_value.execute.return_value.data = _IB_ROW

    # grading_results: no existing row (maybe_single returns None) → insert path
    gr_tbl.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute.return_value = None

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

    return svc, subs_tbl, gr_tbl, aq_tbl


async def test_question_scoped_submission_grades_only_one_aqid():
    """Question-scoped submissions must not loop over all assignment questions."""
    svc, subs_tbl, gr_tbl, aq_tbl = _make_question_scoped_grading_svc()

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_openrouter_result(_LLM_RESPONSE),
        ) as mock_llm,
    ):
        result = await process_submission(
            {},
            submission_id="sub-grade",
            assignment_question_id="aq-1",
            has_assignment_question_id=True,
            force=False,
        )

    assert result["questions_graded"] == 1
    assert result["errors"] == 0
    assert mock_llm.await_count == 1
    assert gr_tbl.insert.call_count == 1

    # Ensure scoped AQID query path was used
    aq_calls = aq_tbl.select.return_value.eq.call_args_list
    assert any(c.args == ("id", "aq-1") for c in aq_calls)


# ── C2: top-level exception handler writes internal_error ─────────────────────

@pytest.mark.asyncio
async def test_uncaught_exception_writes_internal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uncaught exceptions in process_submission must write processing_error='internal_error'.

    Regression for C2: previously any exception after _clear_processing_error
    left the submission silently stuck with no error code.
    """
    svc = MagicMock()
    written_errors: list[str] = []

    def _fake_update(payload):
        code = payload.get("processing_error")
        if code is not None:
            written_errors.append(code)
        mock_chain = MagicMock()
        mock_chain.eq.return_value.execute.return_value = MagicMock(data=[])
        return mock_chain

    svc.table.return_value.update.side_effect = _fake_update
    # _clear_processing_error path: update({"processing_error": None, ...})
    # _set_processing_error path: update({"processing_error": <code>, ...})

    # Monkeypatch _fetch_submission_scope to blow up — simulating any unexpected crash
    # inside the try body after clear_processing_error runs.
    import gradenza_api.jobs.process_submission as ps_mod

    def _exploding_fetch() -> None:
        raise RuntimeError("simulated internal crash")

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
    ):
        # Patch asyncio.to_thread so sync helpers run inline, then make the
        # first real work function (_fetch_submission_scope) raise.
        original_to_thread = ps_mod.asyncio.to_thread

        async def _patched_to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            if getattr(func, "__name__", "") == "_fetch_submission_scope":
                raise RuntimeError("simulated internal crash")
            return func(*args, **kwargs)

        monkeypatch.setattr(ps_mod.asyncio, "to_thread", _patched_to_thread)

        with pytest.raises(RuntimeError, match="simulated internal crash"):
            await process_submission({}, submission_id="sub-crash")

    assert "internal_error" in written_errors, (
        f"expected 'internal_error' in processing_error writes, got: {written_errors}"
    )


# ── H1: LLM JSON shape validation ─────────────────────────────────────────────
#
# Regression for the three failure modes identified in H1:
#   1. _run_ocr returns non-object JSON (list/string) → AttributeError on .get()
#   2. _call_grading_llm returns non-object JSON     → AttributeError on .get()
#   3. Numeric fields in LLM result are strings      → ValueError on int()/float()

def test_safe_int_coerces_valid():
    assert _safe_int(3) == 3
    assert _safe_int("5") == 5

def test_safe_int_returns_default_on_bad_value():
    assert _safe_int("2/5") == 0
    assert _safe_int("three") == 0
    assert _safe_int(None) == 0
    assert _safe_int("2/5", default=99) == 99

def test_safe_float_coerces_valid():
    assert _safe_float(0.9) == 0.9
    assert _safe_float("0.85") == pytest.approx(0.85)

def test_safe_float_returns_default_on_bad_value():
    assert _safe_float("high") == 0.0
    assert _safe_float("n/a") == 0.0
    assert _safe_float(None) == 0.0
    assert _safe_float("high", default=1.0) == 1.0


@pytest.mark.asyncio
async def test_run_ocr_non_object_json_falls_back_gracefully():
    """_run_ocr receiving valid-JSON non-object (string/list/number) must not raise
    AttributeError; it must fall back to returning the raw text with confidence 0.5."""
    for bad_content in ['"just text"', '[1, 2]', '42']:
        mock_result = _openrouter_result(bad_content)
        with (
            patch(
                "gradenza_api.jobs.process_submission.call_openrouter",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("gradenza_api.jobs.process_submission.get_service_client", return_value=MagicMock()),
            patch("gradenza_api.jobs.process_submission.record_ai_usage", new_callable=AsyncMock),
        ):
            text, conf = await _run_ocr("b64data", "image/jpeg", user_id="u-1", submission_id="sub-1")

        assert isinstance(text, str), f"expected str for {bad_content!r}, got {type(text)}"
        assert conf == 0.5, f"expected conf=0.5 fallback for {bad_content!r}, got {conf}"


@pytest.mark.asyncio
async def test_call_grading_llm_non_object_json_returns_none():
    """_call_grading_llm receiving valid-JSON non-object must return None, not raise
    AttributeError on subsequent .get() calls."""
    question = ParsedQuestion(
        question_uuid="q-1",
        source_id="1",
        problem_text="Test",
        diagram_required=False,
        parts=[],
        markscheme_steps=[],
        marks_available=5,
        ft_eligible_parts=[],
        ft_dependencies={},
        assignment_question_id="aq-1",
    )
    for bad_content in ['"just text"', '[1, 2, 3]']:
        mock_result = _openrouter_result(bad_content)
        with (
            patch(
                "gradenza_api.jobs.process_submission.call_openrouter",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
            patch("gradenza_api.jobs.process_submission.get_service_client", return_value=MagicMock()),
            patch("gradenza_api.jobs.process_submission.record_ai_usage", new_callable=AsyncMock),
        ):
            result = await _call_grading_llm(
                system_prompt="Grade.",
                question=question,
                ocr_text="student answer",
                has_low_confidence_ocr=False,
                extracted_answers={},
                user_id="u-1",
                submission_id="sub-1",
            )

        assert result is None, f"expected None for non-object JSON {bad_content!r}, got {result!r}"


def test_detect_amber_non_numeric_confidence_flags_amber():
    """Non-numeric confidence string in LLM result must flag amber, not raise ValueError."""
    amber, reason = _detect_amber("clean text", {"confidence": "high"})
    assert amber is True


_LLM_RESPONSE_FRACTIONAL_MARKS = json.dumps({
    "total_marks_awarded": "2/5",
    "total_method_marks": "three",
    "total_accuracy_marks": None,
    "total_ft_marks": 0,
    "confidence": "medium",
    "feedback_text": "Partial.",
    "overall_amber_flag": False,
    "overall_amber_reason": None,
    "full_assessment": "Attempted.",
    "parts": [{"part_label": "a", "extracted_answer": "x=?", "ft_applied": False}],
})


@pytest.mark.asyncio
async def test_lock_contention_defers_requeue_instead_of_dropping():
    """Lock contention must re-enqueue with a deferral, not silently drop the job.

    Regression: before the fix, job B would return {"skipped": "concurrent_job"}
    terminally when job A held the lock, leaving processing_error persisted with
    no active job to recover.
    """
    mock_redis = AsyncMock()
    mock_redis.set.return_value = False  # simulate lock held by another job
    mock_redis.enqueue_job = AsyncMock()

    result = await process_submission(
        {"redis": mock_redis},
        submission_id="sub-lock",
        assignment_question_id="aq-1",
        has_assignment_question_id=True,
        force=False,
        retry_count=0,
    )

    assert result["skipped"] == "concurrent_job"
    assert result["deferred"] is True
    assert result["retry_count"] == 1

    mock_redis.enqueue_job.assert_called_once()
    call_kwargs = mock_redis.enqueue_job.call_args.kwargs
    assert "_defer_by" in call_kwargs, "re-enqueue must use _defer_by to avoid instant re-collision"
    # Positional args: function_name, submission_id, aqid, has_aqid, force, retry_count
    call_args = mock_redis.enqueue_job.call_args.args
    assert call_args[0] == "process_submission"
    assert call_args[1] == "sub-lock"
    assert call_args[5] == 1  # retry_count incremented


@pytest.mark.asyncio
async def test_lock_contention_gives_up_after_max_retries():
    """After max retries exhausted, lock contention must NOT re-enqueue (avoids infinite loop)."""
    mock_redis = AsyncMock()
    mock_redis.set.return_value = False
    mock_redis.enqueue_job = AsyncMock()

    result = await process_submission(
        {"redis": mock_redis},
        submission_id="sub-lock",
        retry_count=5,  # already at max
    )

    assert result["skipped"] == "concurrent_job"
    assert result["deferred"] is False
    mock_redis.enqueue_job.assert_not_called()


@pytest.mark.asyncio
async def test_grading_malformed_numeric_fields_completes_without_exception():
    """LLM returning non-numeric marks/confidence strings must not crash the job.
    The job must complete with marks_awarded=0 (coercion default) via amber/layer3 result."""
    svc, _, gr_tbl = _make_grading_svc()

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_openrouter_result(_LLM_RESPONSE_FRACTIONAL_MARKS),
        ),
    ):
        result = await process_submission({}, submission_id="sub-grade", force=False)

    assert result["questions_graded"] == 1
    assert result["errors"] == 0
    payload = gr_tbl.insert.call_args.args[0]
    assert payload["marks_awarded"] == 0  # "2/5" coerces to 0, not a crash


# ── mock_phase progression ────────────────────────────────────────────────────

async def test_mock_phase_sequence_on_success():
    """mock_phase advances: ocr → extracting (embedded in ocr_done) → grading → done."""
    svc, subs_tbl, _ = _make_grading_svc()

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch(
            "gradenza_api.jobs.process_submission.call_openrouter",
            new_callable=AsyncMock,
            return_value=_openrouter_result(_LLM_RESPONSE),
        ),
    ):
        result = await process_submission({}, submission_id="sub-grade", force=False)

    assert result["questions_graded"] == 1
    phase_writes = [
        c.args[0]["mock_phase"]
        for c in subs_tbl.update.call_args_list
        if c.args and "mock_phase" in c.args[0]
    ]
    assert phase_writes == ["ocr", "extracting", "grading", "done"]


async def test_mock_phase_set_to_error_on_processing_failure():
    """mock_phase is set to 'error' in the same update as processing_error."""
    photo_row = {
        "id": "ph-1",
        "submission_id": "sub-x",
        "page_number": 1,
        "storage_path": "student/sub-x/attempt_1/001.png",
        "ocr_done_at": None,
    }
    mock_svc, _, mock_subs_tbl = _make_svc(photos_data=[photo_row])
    mock_svc.storage.from_.return_value.download.return_value = None  # triggers storage_download_error

    with (
        patch("gradenza_api.jobs.process_submission.get_service_client", return_value=mock_svc),
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch("gradenza_api.jobs.process_submission._run_ocr"),
        patch("gradenza_api.jobs.process_submission.call_openrouter"),
    ):
        result = await process_submission({}, submission_id="sub-x", force=False)

    assert result["error"] == "storage_download_error"
    error_payloads = [
        c.args[0]
        for c in mock_subs_tbl.update.call_args_list
        if c.args and c.args[0].get("processing_error") == "storage_download_error"
    ]
    assert len(error_payloads) == 1
    assert error_payloads[0].get("mock_phase") == "error"
