"""Endpoint tests for PATCH /v1/quiz/sessions/{sid}/questions/{qid}.

Validates that _validate_mc_options() is enforced at the HTTP boundary and that
_normalize_mc_answer() uppercases correct_answer before the DB write.

No real database or network calls — DB helpers are stubbed via monkeypatching.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import gradenza_api.routes.quiz as quiz_module
from gradenza_api.auth import AuthUser, get_auth_user
from gradenza_api.main import app

_SESSION_ID = "sess-aaa"
_QUESTION_ID = "q-bbb"
_TEACHER_ID = "teacher-ccc"

_EXISTING_MCQ: dict = {
    "id": _QUESTION_ID,
    "session_id": _SESSION_ID,
    "teacher_approved": False,
    "slot": 1,
    "question_json": {
        "question_type": "multiple_choice",
        "options": ["A. Paris", "B. London", "C. Berlin", "D. Madrid"],
        "correct_answer": "A",
        "question_text": "Capital of France?",
        "marks": 1,
        "difficulty": "easy",
        "bloom_level": "remember",
        "explanation": "Paris is the capital of France.",
        "answer_space": "multiple_choice",
    },
}

_PATCH_URL = f"/v1/quiz/sessions/{_SESSION_ID}/questions/{_QUESTION_ID}"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with:
    - auth bypassed (fake teacher injected via get_auth_user override)
    - asyncio.to_thread made synchronous so patched sync helpers are called directly
    - _require_question stubbed to return _EXISTING_MCQ (no DB read)
    - get_service_client stubbed with a chainable mock (no DB write)
    """
    fake_user = AuthUser(id=_TEACHER_ID, role="teacher")
    app.dependency_overrides[get_auth_user] = lambda: fake_user

    async def _sync_to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    monkeypatch.setattr(quiz_module.asyncio, "to_thread", _sync_to_thread)
    monkeypatch.setattr(quiz_module, "_require_question", lambda *_: _EXISTING_MCQ)

    # Chainable mock: .table().update().eq().execute() → returns one updated row
    mock_svc = MagicMock()
    mock_svc.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
        {**_EXISTING_MCQ}
    ]
    monkeypatch.setattr(quiz_module, "get_service_client", lambda: mock_svc)

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


# ── 422 on invalid MCQ edits ──────────────────────────────────────────────────

def test_patch_mcq_multi_char_answer_returns_422(client: TestClient) -> None:
    resp = client.patch(_PATCH_URL, json={"question_json": {"correct_answer": "AB"}})
    assert resp.status_code == 422, resp.text


def test_patch_mcq_duplicate_option_label_returns_422(client: TestClient) -> None:
    resp = client.patch(
        _PATCH_URL,
        json={"question_json": {"options": ["A. One", "A. Two", "B. Three"]}},
    )
    assert resp.status_code == 422, resp.text


def test_patch_mcq_missing_option_prefix_returns_422(client: TestClient) -> None:
    resp = client.patch(
        _PATCH_URL,
        json={"question_json": {"options": ["Paris", "B. London", "C. Berlin"]}},
    )
    assert resp.status_code == 422, resp.text


def test_patch_mcq_too_few_options_returns_422(client: TestClient) -> None:
    resp = client.patch(
        _PATCH_URL,
        json={"question_json": {"options": ["A. Only one"]}},
    )
    assert resp.status_code == 422, resp.text


def test_patch_mcq_answer_not_in_options_returns_422(client: TestClient) -> None:
    resp = client.patch(
        _PATCH_URL,
        json={"question_json": {"correct_answer": "Z"}},
    )
    assert resp.status_code == 422, resp.text


# ── normalization on valid edit ───────────────────────────────────────────────

def test_patch_mcq_lowercase_answer_stored_as_uppercase(client: TestClient) -> None:
    resp = client.patch(_PATCH_URL, json={"question_json": {"correct_answer": "b"}})
    assert resp.status_code == 200, resp.text

    # Inspect what was actually written to the DB
    written_qj = quiz_module.get_service_client().table().update.call_args[0][0]["question_json"]
    assert written_qj["correct_answer"] == "B"


def test_patch_mcq_valid_edit_returns_200(client: TestClient) -> None:
    resp = client.patch(
        _PATCH_URL,
        json={"question_json": {"question_text": "Updated question text?"}},
    )
    assert resp.status_code == 200, resp.text
