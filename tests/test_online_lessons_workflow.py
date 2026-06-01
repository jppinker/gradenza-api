"""Focused tests for the online lesson generation workflow."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

import gradenza_api.routes.online_lessons as online_lessons
from gradenza_api.auth import AuthUser


LESSON_ID = "11111111-1111-4111-8111-111111111111"
TEACHER_ID = "teacher-online-lesson-1"


def _usage(model: str = "google/gemini-2.5-flash") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150
    usage.model = model
    usage.request_id = "req-online-lesson-test"
    return usage


def _result(content: str, model: str = "google/gemini-2.5-flash") -> MagicMock:
    result = MagicMock()
    result.content = content
    result.usage = _usage(model)
    return result


@pytest.fixture(autouse=True)
def _common_route_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    lesson = {
        "id": LESSON_ID,
        "teacher_id": TEACHER_ID,
        "kind": "online",
        "class_id": None,
        "title": "AAHL trig transformations",
        "lesson_plan": (
            "## Lesson Overview\nTeach amplitude, period, and phase shift in trig graphs.\n\n"
            "## Learning Objectives\n- Transform sine graphs."
        ),
        "lesson_plan_status": "approved",
        "lesson_plan_generated_at": "2026-01-01T10:00:00Z",
        "lesson_plan_approved_at": "2026-01-01T10:05:00Z",
        "lesson_plan_source_meta": {"sourcePlanHash": "hash-1", "chatHistoryTurns": 2},
    }

    monkeypatch.setattr(online_lessons, "get_service_client", lambda: MagicMock())
    monkeypatch.setattr(
        online_lessons,
        "_verify_lesson_access",
        AsyncMock(return_value=lesson),
    )
    monkeypatch.setattr(
        online_lessons,
        "_fetch_lesson_context",
        AsyncMock(return_value={"students": [], "className": None, "notesCount": 0, "materialsCount": 0}),
    )
    monkeypatch.setattr(online_lessons, "record_ai_usage", AsyncMock())


@pytest.mark.asyncio
async def test_chat_readiness_true_for_clear_topic_and_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        online_lessons,
        "_call_with_structured_output",
        AsyncMock(return_value=(
            {
                "reply": "You can generate the plan now. Add level or duration if you want a tighter draft.",
                "readyToGenerateLessonPlan": True,
                "missingContext": ["Lesson duration"],
                "detectedTopic": "AA HL trig graph transformations",
                "detectedObjectives": ["Focus on amplitude, period, phase shift, and common mistakes"],
                "suggestedQuestionCount": 5,
            },
            _result("{}"),
        )),
    )

    response = await online_lessons.online_lesson_chat(
        online_lessons.OnlineLessonChatRequest(
            lesson_id=LESSON_ID,
            message="I want to teach AAHL trig graph transformations, focus on amplitude, period, phase shift, and common mistakes",
        ),
        AuthUser(id=TEACHER_ID, role="teacher"),
    )

    assert response.readyToGenerateLessonPlan is True
    assert response.detectedTopic == "AA HL trig graph transformations"
    assert response.missingContext == ["Lesson duration"]
    assert response.suggestedQuestionCount == 5


@pytest.mark.asyncio
async def test_chat_readiness_false_for_vague_topic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        online_lessons,
        "_call_with_structured_output",
        AsyncMock(return_value=(
            {
                "reply": "Which area of calculus - limits, derivatives, integrals, or something specific?",
                "readyToGenerateLessonPlan": False,
                "missingContext": ["Specific calculus concept or teachable goal"],
                "detectedTopic": "",
                "detectedObjectives": [],
                "suggestedQuestionCount": 0,
            },
            _result("{}"),
        )),
    )

    response = await online_lessons.online_lesson_chat(
        online_lessons.OnlineLessonChatRequest(lesson_id=LESSON_ID, message="calculus"),
        AuthUser(id=TEACHER_ID, role="teacher"),
    )

    assert response.readyToGenerateLessonPlan is False
    assert response.detectedTopic == ""
    assert response.suggestedQuestionCount == 0


@pytest.mark.asyncio
async def test_generate_plan_accepts_chat_history_and_returns_draft_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_call = AsyncMock(return_value=_result("## Lesson Overview\nGenerated plan body"))
    monkeypatch.setattr(online_lessons, "call_openrouter", mock_call)

    response = await online_lessons.generate_lesson_plan(
        online_lessons.GeneratePlanRequest(
            lesson_id=LESSON_ID,
            chat_history=[
                online_lessons.ChatHistoryTurn(role="user", content="Teach trig transformations."),
                online_lessons.ChatHistoryTurn(role="assistant", content="Ready to draft now."),
            ],
        ),
        AuthUser(id=TEACHER_ID, role="teacher"),
    )

    assert response.plan.startswith("## Lesson Overview")
    assert response.source_meta["promptVersion"] == "online_lesson_plan_v1"
    assert response.source_meta["chatHistoryTurns"] == 2
    messages = mock_call.await_args.kwargs["messages"]
    assert messages[-3]["content"] == "Teach trig transformations."
    assert messages[-2]["content"] == "Ready to draft now."


def test_practice_question_options_validate_successfully() -> None:
    request = online_lessons.GenerateQuestionsRequest.model_validate({
        "lesson_id": LESSON_ID,
        "generation_options": {
            "question_count": 5,
            "type_mix": {"mcq": 1, "short_answer": 3, "frq": 1},
            "difficulty_mix": {"easy": 1, "medium": 3, "hard": 1},
            "difficulty_definitions": {
                "easy": "Recall one known graph feature.",
                "medium": "Apply graph transformations in a familiar context.",
                "challenge": "Synthesize multiple transformations from an unfamiliar graph.",
            },
            "marks_min": 1,
            "marks_max": 6,
            "total_marks_target": 15,
            "exam_style": "IB AA HL",
            "include_worked_solutions": True,
            "include_markscheme": True,
            "include_tags": True,
            "teacher_instructions": "Include one graph-sketching item.",
        },
    })

    assert request.question_count == 5
    assert request.generation_options is not None
    assert request.generation_options.type_mix is not None
    assert request.generation_options.type_mix.multiple_choice == 1
    assert request.generation_options.difficulty_mix is not None
    assert request.generation_options.difficulty_mix.challenge == 1
    assert request.generation_options.difficulty_definitions.challenge.startswith("Synthesize")


def test_practice_question_options_reject_impossible_mix() -> None:
    with pytest.raises(ValidationError, match="type_mix total"):
        online_lessons.GenerateQuestionsRequest.model_validate({
            "lesson_id": LESSON_ID,
            "question_count": 3,
            "generation_options": {
                "type_mix": {"multiple_choice": 2, "short_answer": 2},
            },
        })


def test_generated_rich_question_json_validates_and_repairs_marks() -> None:
    question = online_lessons._repair_question(
        {
            "slot": 1,
            "title": "Amplitude from a Graph",
            "question_type": "multiple_choice",
            "difficulty": "challenge",
            "marks": 4,
            "estimated_time_minutes": 5,
            "question_text": "A sine graph has maximum 3 and minimum -1. Find the amplitude.",
            "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
            "correct_answer": "B",
            "worked_solution": "Amplitude is half the vertical range.",
            "markscheme_steps": [{"description": "Use half range", "marks": 2}],
            "common_mistakes": ["Using the maximum as amplitude"],
            "hints": ["Compare max and min values."],
            "tags": ["trig", "graphs"],
            "domain": "Trigonometry",
            "topic": "Graph transformations",
            "subtopic": "Amplitude",
            "exam_system": "IB",
            "subject": "Mathematics",
            "level": "HL",
            "source_lesson_plan_excerpt": "Teach amplitude, period, and phase shift in trig graphs.",
            "quality_notes": "Targets a common misconception.",
        },
        slot=1,
    )

    assert question.question_type == "multiple_choice"
    assert sum(step.marks for step in question.markscheme_steps) == question.marks
    assert question.options == ["A. 1", "B. 2", "C. 3", "D. 4"]


@pytest.mark.asyncio
async def test_generate_questions_returns_rich_questions_and_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "questions": [
            {
                "slot": 1,
                "title": "Amplitude from a Graph",
                "question_type": "short_answer",
                "difficulty": "medium",
                "marks": 3,
                "estimated_time_minutes": 4,
                "question_text": "Find the amplitude from max 5 and min -1.",
                "options": [],
                "correct_answer": "3",
                "worked_solution": "Amplitude = (5 - (-1)) / 2 = 3.",
                "markscheme_steps": [{"description": "Correct half-range method", "marks": 3}],
                "common_mistakes": ["Using total range"],
                "hints": ["Amplitude is half the range."],
                "tags": ["trig", "amplitude"],
                "domain": "Trigonometry",
                "topic": "Graph transformations",
                "subtopic": "Amplitude",
                "exam_system": "IB",
                "subject": "Mathematics",
                "level": "HL",
                "source_lesson_plan_excerpt": "Teach amplitude, period, and phase shift in trig graphs.",
                "quality_notes": "Checks conceptual understanding.",
            }
        ]
    }
    mock_call = AsyncMock(return_value=_result(json.dumps(raw)))
    monkeypatch.setattr(online_lessons, "call_openrouter", mock_call)

    response = await online_lessons.generate_practice_questions(
        online_lessons.GenerateQuestionsRequest(
            lesson_id=LESSON_ID,
            question_count=1,
            generation_options=online_lessons.GenerationOptions(
                difficulty_definitions=online_lessons.DifficultyDefinitions(
                    easy="Teacher custom easy definition.",
                    medium="Teacher custom medium definition.",
                    challenge="Teacher custom challenge definition.",
                ),
            ),
        ),
        AuthUser(id=TEACHER_ID, role="teacher"),
    )

    assert len(response.questions) == 1
    assert response.questions[0].source_lesson_id == LESSON_ID
    assert response.source_meta["createdFrom"] == "online-lesson"
    assert response.source_meta["sourcePlanHash"] == "hash-1"
    system_prompt = mock_call.await_args.kwargs["messages"][0]["content"]
    assert "Teacher custom easy definition." in system_prompt
    assert "Teacher custom medium definition." in system_prompt
    assert "Teacher custom challenge definition." in system_prompt
    assert response.source_meta["generation_options"]["difficulty_definitions"]["easy"] == "Teacher custom easy definition."
