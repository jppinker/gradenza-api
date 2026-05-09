"""Tests for POST /v1/canvas/chat.

Focused coverage:
- Route is registered (does not 404) and returns expected shape on success.
- OpenRouter failure path returns 502 + records failed usage.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import gradenza_api.routes.canvas_chat as canvas_chat_module
from gradenza_api.auth import AuthUser, get_auth_user
from gradenza_api.main import app


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    fake_user = AuthUser(id="teacher-canvas-chat-1", role="teacher")
    app.dependency_overrides[get_auth_user] = lambda: fake_user

    mock_svc = MagicMock()
    monkeypatch.setattr(canvas_chat_module, "get_service_client", lambda: mock_svc)
    monkeypatch.setattr(canvas_chat_module, "record_ai_usage", AsyncMock())

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    app.dependency_overrides.clear()


def _make_openrouter_result(content: str) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.model = "google/gemini-2.5-flash"
    usage.request_id = "req-canvas-chat-test-001"
    result = MagicMock()
    result.content = content
    result.usage = usage
    return result


def test_post_canvas_chat_is_registered_and_returns_200(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_call = AsyncMock(return_value=_make_openrouter_result("Canvas answer"))
    monkeypatch.setattr(canvas_chat_module, "call_openrouter", mock_call)

    resp = client.post(
        "/v1/canvas/chat",
        json={
            "canvas_id": "canvas-abc",
            "prompt": "Explain $x^2$ briefly.",
            "conversation": [{"role": "user", "content": "Hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"answer": "Canvas answer"}

    kwargs = mock_call.await_args.kwargs
    assert kwargs["model"] == "google/gemini-2.5-flash"
    msgs = kwargs["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "Hi"
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"] == "Explain $x^2$ briefly."


def test_canvas_chat_openrouter_failure_returns_502_and_records_usage(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_mock = AsyncMock()
    monkeypatch.setattr(canvas_chat_module, "record_ai_usage", record_mock)
    monkeypatch.setattr(canvas_chat_module, "call_openrouter", AsyncMock(side_effect=Exception("boom")))

    resp = client.post(
        "/v1/canvas/chat",
        json={
            "canvas_id": "canvas-err",
            "prompt": "Test",
            "conversation": [],
        },
    )
    assert resp.status_code == 502, resp.text
    assert resp.json().get("detail") == "AI service unavailable"
    assert record_mock.await_count == 1
    assert record_mock.await_args.kwargs["status"] == "error"

