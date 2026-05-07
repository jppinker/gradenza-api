"""Tests for quiz question image attach / storage / export rendering.

Covers:
- PATCH endpoint stores image_slot correctly in question_json
- _generate_quiz_markdown renders image above and below question
- _is_quiz_image_url security scoping
- _render_question_html positions image correctly

No real database or network calls — DB helpers are stubbed via monkeypatching.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import gradenza_api.routes.quiz as quiz_module
from gradenza_api.auth import AuthUser, get_auth_user
from gradenza_api.main import app
from gradenza_api.routes.quiz import (
    _generate_quiz_markdown,
    _is_quiz_image_url,
    _render_question_html,
)

_SESSION_ID = "sess-img-test"
_QUESTION_ID = "q-img-test"
_TEACHER_ID = "teacher-img-test"
_SUPABASE_URL = "https://abc.supabase.co"

_BASE_QUESTION: dict = {
    "id": _QUESTION_ID,
    "session_id": _SESSION_ID,
    "teacher_approved": False,
    "slot": 1,
    "question_json": {
        "question_type": "short_answer",
        "question_text": "Explain photosynthesis.",
        "correct_answer": "Light energy converts CO2 and water into glucose.",
        "marks": 2,
        "difficulty": "medium",
        "explanation": "",
        "answer_space": "medium",
        "image_slot": None,
    },
}

_PATCH_URL = f"/v1/quiz/sessions/{_SESSION_ID}/questions/{_QUESTION_ID}"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_user = AuthUser(id=_TEACHER_ID, role="teacher")
    app.dependency_overrides[get_auth_user] = lambda: fake_user

    async def _sync_to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    monkeypatch.setattr(quiz_module.asyncio, "to_thread", _sync_to_thread)
    monkeypatch.setattr(quiz_module, "_require_question", lambda *_: _BASE_QUESTION)

    mock_svc = MagicMock()
    mock_svc.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
        {**_BASE_QUESTION}
    ]
    monkeypatch.setattr(quiz_module, "get_service_client", lambda: mock_svc)

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


# ── PATCH stores image_slot ───────────────────────────────────────────────────

def test_patch_image_slot_stored_in_question_json(client: TestClient) -> None:
    image_slot = {
        "url": f"{_SUPABASE_URL}/storage/v1/object/public/quiz-images/{_SESSION_ID}/{_QUESTION_ID}/abc.png",
        "storage_path": f"{_SESSION_ID}/{_QUESTION_ID}/abc.png",
        "position": "above",
        "caption": "Graph of f(x)",
        "size": "medium",
    }
    resp = client.patch(_PATCH_URL, json={"question_json": {"image_slot": image_slot}})
    assert resp.status_code == 200, resp.text

    written_qj = quiz_module.get_service_client().table().update.call_args[0][0]["question_json"]
    stored_slot = written_qj.get("image_slot")
    assert stored_slot is not None
    assert stored_slot["url"] == image_slot["url"]
    assert stored_slot["storage_path"] == image_slot["storage_path"]
    assert stored_slot["position"] == "above"
    assert stored_slot["caption"] == "Graph of f(x)"


def test_patch_image_slot_null_clears_image(client: TestClient) -> None:
    resp = client.patch(_PATCH_URL, json={"question_json": {"image_slot": None}})
    assert resp.status_code == 200, resp.text

    written_qj = quiz_module.get_service_client().table().update.call_args[0][0]["question_json"]
    assert written_qj.get("image_slot") is None


# ── Markdown export includes images ──────────────────────────────────────────

def _make_session(title: str = "Test Quiz") -> dict:
    return {"id": _SESSION_ID, "title": title, "subject": "Biology"}


def _make_question(
    position: str,
    caption: str = "",
    url: str | None = None,
    question_id: str = _QUESTION_ID,
) -> dict:
    img_url = url if url is not None else _quiz_img_url(question_id=question_id)
    return {
        "id": question_id,
        "slot": 1,
        "question_json": {
            "question_type": "short_answer",
            "question_text": "What is osmosis?",
            "correct_answer": "Movement of water across a semi-permeable membrane.",
            "marks": 1,
            "explanation": "",
            "image_slot": {
                "url": img_url,
                "storage_path": f"{_SESSION_ID}/{question_id}/graph.png",
                "position": position,
                "caption": caption,
                "size": "medium",
            },
        },
    }


def test_markdown_image_above_appears_before_answer_blank() -> None:
    md = _generate_quiz_markdown(_make_session(), [_make_question("above", "Fig 1")])
    img_idx = md.index(f"![Fig 1]({_quiz_img_url()})")
    blank_idx = md.index("_" * 60)
    assert img_idx < blank_idx, "Image should appear before the answer blank when position=above"


def test_markdown_image_below_appears_after_answer_blank() -> None:
    md = _generate_quiz_markdown(_make_session(), [_make_question("below", "Fig 2")])
    img_idx = md.index(f"![Fig 2]({_quiz_img_url()})")
    blank_idx = md.index("_" * 60)
    assert img_idx > blank_idx, "Image should appear after the answer blank when position=below"


def test_markdown_image_empty_caption_uses_empty_alt() -> None:
    md = _generate_quiz_markdown(_make_session(), [_make_question("above", "")])
    assert f"![]({_quiz_img_url()})" in md


def test_markdown_no_image_slot_omits_image_markdown() -> None:
    q = {"id": _QUESTION_ID, "slot": 1, "question_json": {"question_type": "short_answer", "question_text": "No image.", "marks": 1, "correct_answer": "x", "explanation": "", "image_slot": None}}
    md = _generate_quiz_markdown(_make_session(), [q])
    assert "![" not in md


def test_markdown_teacher_version_includes_answer_with_image() -> None:
    md = _generate_quiz_markdown(_make_session(), [_make_question("above")], version="teacher")
    assert "**Answer:**" in md
    assert "![" in md


def test_markdown_external_image_url_omitted() -> None:
    """Markdown must not emit images from URLs that fail the quiz-image scoping check."""
    q = _make_question("above", "Evil image", url="https://evil.com/evil.png")
    md = _generate_quiz_markdown(_make_session(), [q])
    assert "![" not in md, "External/unscoped image URLs must be suppressed in markdown export"


# ── _is_quiz_image_url security scoping ──────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_supabase_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quiz_module.settings, "supabase_url", _SUPABASE_URL)


def _quiz_img_url(session_id: str = _SESSION_ID, question_id: str = _QUESTION_ID) -> str:
    return f"{_SUPABASE_URL}/storage/v1/object/public/quiz-images/{session_id}/{question_id}/img.png"


def test_is_quiz_image_url_accepts_own_question() -> None:
    assert _is_quiz_image_url(_quiz_img_url(), _SESSION_ID, _QUESTION_ID)


def test_is_quiz_image_url_rejects_different_session() -> None:
    assert not _is_quiz_image_url(_quiz_img_url(session_id="other-session"), _SESSION_ID, _QUESTION_ID)


def test_is_quiz_image_url_rejects_different_question() -> None:
    assert not _is_quiz_image_url(_quiz_img_url(question_id="other-question"), _SESSION_ID, _QUESTION_ID)


def test_is_quiz_image_url_rejects_external_url() -> None:
    assert not _is_quiz_image_url("https://evil.com/evil.png", _SESSION_ID, _QUESTION_ID)


# ── _render_question_html positions image correctly ──────────────────────────

def _render(position: str, img_url: str | None = None) -> str:
    qj = {
        "question_type": "short_answer",
        "question_text": "Define osmosis.",
        "marks": 1,
        "difficulty": "easy",
        "correct_answer": "x",
        "explanation": "",
        "answer_space": "small",
        "image_slot": {
            "url": img_url or _quiz_img_url(),
            "storage_path": f"{_SESSION_ID}/{_QUESTION_ID}/img.png",
            "position": position,
            "caption": "Test caption",
            "size": "medium",
        } if img_url is not False else None,
    }
    return _render_question_html(1, qj, "small", True, False, session_id=_SESSION_ID, question_id=_QUESTION_ID)


def test_render_html_image_above_precedes_question_text() -> None:
    html = _render("above")
    img_idx = html.index("question-image")
    text_idx = html.index("question-text")
    assert img_idx < text_idx


def test_render_html_image_below_follows_question_text() -> None:
    html = _render("below")
    img_idx = html.index("question-image")
    text_idx = html.index("question-text")
    assert img_idx > text_idx
