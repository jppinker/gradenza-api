"""Tests for POST /v1/quiz/sessions/{sid}/questions/{qid}/regenerate.

Covers:
- clarifying_prompt accepted in request body
- body omitted (backward compat)
- structured JSON retry path records failed attempt usage
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import gradenza_api.routes.quiz as quiz_module
from gradenza_api.auth import AuthUser, get_auth_user
from gradenza_api.main import app
from gradenza_api.services.llm_structured import (
    StructuredAttempt,
    StructuredJSONError,
    StructuredJSONResult,
)

_SESSION_ID = "sess-regen-aaa"
_QUESTION_ID = "q-regen-bbb"
_TEACHER_ID = "teacher-regen-ccc"

_EXISTING_QUESTION: dict = {
    "id": _QUESTION_ID,
    "session_id": _SESSION_ID,
    "slot": 1,
    "teacher_approved": False,
    "validation_status": "passed",
    "validation_warnings": [],
    "question_json": {
        "slot": 1,
        "question_type": "short_answer",
        "difficulty": "medium",
        "bloom_level": "understand",
        "question_text": "Explain photosynthesis.",
        "options": None,
        "correct_answer": "Plants convert sunlight into glucose.",
        "acceptable_answers": None,
        "explanation": "Photosynthesis is the process by which plants make food.",
        "marks": 2,
        "rubric": None,
        "answer_space": "short_paragraph",
        "source_reference": None,
        "image_slot": None,
    },
}

_EXISTING_SESSION: dict = {
    "id": _SESSION_ID,
    "teacher_id": _TEACHER_ID,
    "mode": "from_source",
    "preset": "standard_worksheet",
    "subject": "Biology",
    "grade_level": "Grade 10",
    "status": "review",
}

_REGENERATED_QJ: dict = {
    "slot": 1,
    "question_type": "short_answer",
    "difficulty": "medium",
    "bloom_level": "apply",
    "question_text": "Describe the role of chlorophyll in photosynthesis.",
    "options": None,
    "correct_answer": "Chlorophyll absorbs light energy.",
    "acceptable_answers": None,
    "explanation": "Chlorophyll is the pigment that captures light.",
    "marks": 2,
    "rubric": None,
    "answer_space": "short_paragraph",
    "source_reference": None,
    "image_slot": None,
}

_REGEN_URL = f"/v1/quiz/sessions/{_SESSION_ID}/questions/{_QUESTION_ID}/regenerate"


def _make_llm_result() -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150
    usage.model = "google/gemini-2.5-flash"
    usage.request_id = "req-test-001"
    result = MagicMock()
    result.usage = usage
    result.content = "{}"
    return result


def _make_structured_result(qj: dict) -> StructuredJSONResult:
    llm_result = _make_llm_result()
    attempt = StructuredAttempt(
        attempt=1,
        llm_result=llm_result,
        raw="{}",
        status="success",
    )
    return StructuredJSONResult(
        value=qj,
        llm_result=llm_result,
        attempt=1,
        raw="{}",
        attempts=[attempt],
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_user = AuthUser(id=_TEACHER_ID, role="teacher")
    app.dependency_overrides[get_auth_user] = lambda: fake_user

    async def _sync_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(quiz_module.asyncio, "to_thread", _sync_to_thread)
    monkeypatch.setattr(quiz_module, "_require_question", lambda *_: _EXISTING_QUESTION)
    monkeypatch.setattr(quiz_module, "_require_session", lambda *_: _EXISTING_SESSION)

    # DB mocks: source/blueprint return None (no source material), update returns updated row
    mock_svc = MagicMock()
    mock_svc.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = []
    updated_row = {**_EXISTING_QUESTION, "question_json": _REGENERATED_QJ, "teacher_approved": False}
    mock_svc.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [updated_row]
    monkeypatch.setattr(quiz_module, "get_service_client", lambda: mock_svc)

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


# ── clarifying_prompt in body ─────────────────────────────────────────────────

def test_regenerate_with_clarifying_prompt(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured_prompt: list[str] = []

    async def _mock_structured(**kwargs):
        # Capture the prompt from base_messages
        user_msg = next((m["content"] for m in kwargs["base_messages"] if m["role"] == "user"), "")
        captured_prompt.append(user_msg)
        return _make_structured_result(_REGENERATED_QJ)

    monkeypatch.setattr(quiz_module, "call_openrouter_structured_json", _mock_structured)
    monkeypatch.setattr(quiz_module, "record_ai_usage", AsyncMock())

    resp = client.post(_REGEN_URL, json={"clarifying_prompt": "Make it harder"})
    assert resp.status_code == 200, resp.text
    assert len(captured_prompt) == 1
    assert "Make it harder" in captured_prompt[0]


def test_regenerate_clarifying_prompt_trimmed_and_limited(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    captured_prompt: list[str] = []

    async def _mock_structured(**kwargs):
        user_msg = next((m["content"] for m in kwargs["base_messages"] if m["role"] == "user"), "")
        captured_prompt.append(user_msg)
        return _make_structured_result(_REGENERATED_QJ)

    monkeypatch.setattr(quiz_module, "call_openrouter_structured_json", _mock_structured)
    monkeypatch.setattr(quiz_module, "record_ai_usage", AsyncMock())

    long_prompt = "A" * 2000
    resp = client.post(_REGEN_URL, json={"clarifying_prompt": long_prompt})
    assert resp.status_code == 200, resp.text
    assert "A" * 1000 in captured_prompt[0]
    assert "A" * 1001 not in captured_prompt[0]


# ── body omitted (backward compat) ───────────────────────────────────────────

def test_regenerate_no_body(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_structured(**kwargs):
        user_msg = next((m["content"] for m in kwargs["base_messages"] if m["role"] == "user"), "")
        # no clarifying prompt section should appear
        assert "Teacher's instruction:" not in user_msg
        return _make_structured_result(_REGENERATED_QJ)

    monkeypatch.setattr(quiz_module, "call_openrouter_structured_json", _mock_structured)
    monkeypatch.setattr(quiz_module, "record_ai_usage", AsyncMock())

    resp = client.post(_REGEN_URL)
    assert resp.status_code == 200, resp.text


def test_regenerate_empty_clarifying_prompt_treated_as_none(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_structured(**kwargs):
        user_msg = next((m["content"] for m in kwargs["base_messages"] if m["role"] == "user"), "")
        assert "Teacher's instruction:" not in user_msg
        return _make_structured_result(_REGENERATED_QJ)

    monkeypatch.setattr(quiz_module, "call_openrouter_structured_json", _mock_structured)
    monkeypatch.setattr(quiz_module, "record_ai_usage", AsyncMock())

    resp = client.post(_REGEN_URL, json={"clarifying_prompt": "   "})
    assert resp.status_code == 200, resp.text


# ── structured JSON retry / usage recording ───────────────────────────────────

def test_regenerate_retry_path_records_failed_attempt_usage(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """StructuredJSONError → all failed attempts recorded → 502 returned."""
    failed_llm = _make_llm_result()
    failed_att = StructuredAttempt(
        attempt=1,
        llm_result=failed_llm,
        raw="not json",
        status="json_parse_error",
        error_message="Expecting value: line 1 column 1",
    )
    failed_att2 = StructuredAttempt(
        attempt=2,
        llm_result=_make_llm_result(),
        raw="still not json",
        status="json_parse_error",
        error_message="Expecting value: line 1 column 1",
    )
    exc = StructuredJSONError(
        "failed after 2 attempts",
        attempts=[failed_att, failed_att2],
        last_exc=ValueError("parse error"),
    )

    async def _mock_structured(**kwargs):
        raise exc

    usage_calls: list[dict] = []

    async def _mock_record_usage(svc, *, status="success", error_message=None, **kwargs):
        usage_calls.append({"status": status, "error_message": error_message})

    monkeypatch.setattr(quiz_module, "call_openrouter_structured_json", _mock_structured)
    monkeypatch.setattr(quiz_module, "record_ai_usage", _mock_record_usage)

    resp = client.post(_REGEN_URL)
    assert resp.status_code == 502, resp.text

    # Both failed attempts must be recorded with status="error"
    assert len(usage_calls) == 2
    assert all(c["status"] == "error" for c in usage_calls)
    assert all(c["error_message"] for c in usage_calls)


def test_regenerate_success_records_usage_with_correct_status(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    structured = _make_structured_result(_REGENERATED_QJ)

    async def _mock_structured(**kwargs):
        return structured

    usage_calls: list[dict] = []

    async def _mock_record_usage(svc, *, status="success", error_message=None, **kwargs):
        usage_calls.append({"status": status, "error_message": error_message})

    monkeypatch.setattr(quiz_module, "call_openrouter_structured_json", _mock_structured)
    monkeypatch.setattr(quiz_module, "record_ai_usage", _mock_record_usage)

    resp = client.post(_REGEN_URL, json={"clarifying_prompt": "Use a diagram"})
    assert resp.status_code == 200, resp.text
    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == "success"


def test_regenerate_uses_structured_json_not_raw_call(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure call_openrouter_structured_json is used (not the old direct call_openrouter path)."""
    structured_called = []

    async def _mock_structured(**kwargs):
        structured_called.append(True)
        return _make_structured_result(_REGENERATED_QJ)

    monkeypatch.setattr(quiz_module, "call_openrouter_structured_json", _mock_structured)
    monkeypatch.setattr(quiz_module, "record_ai_usage", AsyncMock())

    resp = client.post(_REGEN_URL)
    assert resp.status_code == 200, resp.text
    assert structured_called, "call_openrouter_structured_json was not called"
