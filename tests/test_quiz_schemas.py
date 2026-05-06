"""Unit tests for QuizQuestionSchema.

No database, network, or environment variables required.
"""

import pytest
from pydantic import ValidationError

from gradenza_api.services.quiz_schemas import QuizQuestionSchema


def _base(**overrides) -> dict:
    """Minimal valid short_answer question."""
    q = {
        "slot": 1,
        "question_type": "short_answer",
        "difficulty": "medium",
        "bloom_level": "understand",
        "question_text": "What is photosynthesis?",
        "options": None,
        "correct_answer": "The process plants use to convert light to energy",
        "acceptable_answers": None,
        "explanation": "Photosynthesis converts CO2 + H2O into glucose using sunlight.",
        "marks": 2,
        "rubric": None,
        "answer_space": "short_paragraph",
        "source_reference": None,
        "image_slot": None,
    }
    q.update(overrides)
    return q


def _mc(**overrides) -> dict:
    """Minimal valid multiple_choice question."""
    q = _base(
        question_type="multiple_choice",
        options=["A. Paris", "B. London", "C. Berlin", "D. Madrid"],
        correct_answer="A",
        answer_space="multiple_choice",
    )
    q.update(overrides)
    return q


# ── difficulty ────────────────────────────────────────────────────────────────

def test_difficulty_easy():
    QuizQuestionSchema.model_validate(_base(difficulty="easy"))


def test_difficulty_medium():
    QuizQuestionSchema.model_validate(_base(difficulty="medium"))


def test_difficulty_challenge():
    QuizQuestionSchema.model_validate(_base(difficulty="challenge"))


def test_difficulty_hard_rejected():
    with pytest.raises(ValidationError, match="difficulty"):
        QuizQuestionSchema.model_validate(_base(difficulty="hard"))


def test_difficulty_missing_rejected():
    data = _base()
    del data["difficulty"]
    with pytest.raises(ValidationError):
        QuizQuestionSchema.model_validate(data)


# ── bloom_level ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", ["remember", "understand", "apply", "analyse", "evaluate", "create"])
def test_bloom_level_valid(level: str):
    QuizQuestionSchema.model_validate(_base(bloom_level=level))


def test_bloom_level_invalid_rejected():
    with pytest.raises(ValidationError, match="bloom_level"):
        QuizQuestionSchema.model_validate(_base(bloom_level="synthesize"))


# ── slot / marks bounds ───────────────────────────────────────────────────────

def test_slot_zero_rejected():
    with pytest.raises(ValidationError, match="slot"):
        QuizQuestionSchema.model_validate(_base(slot=0))


def test_marks_zero_rejected():
    with pytest.raises(ValidationError, match="marks"):
        QuizQuestionSchema.model_validate(_base(marks=0))


# ── multiple_choice shape ─────────────────────────────────────────────────────

def test_mc_valid():
    QuizQuestionSchema.model_validate(_mc())


def test_mc_answer_b():
    QuizQuestionSchema.model_validate(_mc(correct_answer="B"))


def test_mc_answer_with_dot():
    # "A." should be accepted — trailing dot stripped during normalization
    validated = QuizQuestionSchema.model_validate(_mc(correct_answer="A."))
    assert validated.correct_answer == "A"


def test_mc_no_options_rejected():
    with pytest.raises(ValidationError, match="at least 2 options"):
        QuizQuestionSchema.model_validate(_mc(options=None))


def test_mc_empty_options_rejected():
    with pytest.raises(ValidationError, match="at least 2 options"):
        QuizQuestionSchema.model_validate(_mc(options=[]))


def test_mc_single_option_rejected():
    with pytest.raises(ValidationError, match="at least 2 options"):
        QuizQuestionSchema.model_validate(_mc(options=["A. Only option"]))


def test_mc_option_without_letter_prefix_rejected():
    with pytest.raises(ValidationError, match="must start with a letter and dot"):
        QuizQuestionSchema.model_validate(_mc(options=["Paris", "B. London", "C. Berlin", "D. Madrid"]))


def test_mc_duplicate_option_label_rejected():
    with pytest.raises(ValidationError, match="duplicate option label"):
        QuizQuestionSchema.model_validate(_mc(options=["A. One", "A. Two", "B. Three", "C. Four"]))


def test_mc_correct_answer_not_in_options_rejected():
    with pytest.raises(ValidationError, match="correct_answer"):
        QuizQuestionSchema.model_validate(_mc(correct_answer="Z"))


def test_mc_correct_answer_multi_char_rejected():
    with pytest.raises(ValidationError, match="correct_answer"):
        QuizQuestionSchema.model_validate(_mc(correct_answer="AB"))


def test_mc_correct_answer_lowercase_normalized():
    # Lowercase "a" must be accepted and stored as uppercase "A"
    validated = QuizQuestionSchema.model_validate(_mc(correct_answer="a"))
    assert validated.correct_answer == "A"


def test_mc_correct_answer_stored_normalized_in_dump():
    # model_dump() must reflect the canonical uppercase letter, not the raw input
    dumped = QuizQuestionSchema.model_validate(_mc(correct_answer="a")).model_dump()
    assert dumped["correct_answer"] == "A"

    dumped2 = QuizQuestionSchema.model_validate(_mc(correct_answer="A.")).model_dump()
    assert dumped2["correct_answer"] == "A"


# ── non-MC questions ignore options rules ─────────────────────────────────────

def test_short_answer_no_options_ok():
    QuizQuestionSchema.model_validate(_base(options=None))


def test_fill_blank_no_options_ok():
    QuizQuestionSchema.model_validate(_base(
        question_type="fill_blank",
        answer_space="single_line",
        options=None,
    ))


# ── image_slot ────────────────────────────────────────────────────────────────

def test_image_slot_none_ok():
    QuizQuestionSchema.model_validate(_base(image_slot=None))


def test_image_slot_string_rejected():
    # LLM may return a stray string; schema must reject it so render code never crashes
    with pytest.raises(ValidationError):
        QuizQuestionSchema.model_validate(_base(image_slot="https://example.com/img.png"))


def test_image_slot_dict_rejected():
    # Populated image_slot objects are set by the image-upload flow, not LLM generation
    with pytest.raises(ValidationError):
        QuizQuestionSchema.model_validate(_base(image_slot={"url": "https://example.com/img.png"}))


# ── model_dump output ─────────────────────────────────────────────────────────

def test_model_dump_contains_expected_keys():
    validated = QuizQuestionSchema.model_validate(_mc())
    dumped = validated.model_dump()
    assert dumped["difficulty"] == "medium"
    assert dumped["question_type"] == "multiple_choice"
    assert dumped["correct_answer"] == "A"
    assert dumped["slot"] == 1


def test_model_dump_preserves_extra_field():
    # extra="allow" — LLM-added fields must survive the round-trip
    data = _base(some_llm_extra_field="foo")
    dumped = QuizQuestionSchema.model_validate(data).model_dump()
    assert dumped.get("some_llm_extra_field") == "foo"


# ── _normalize_mc_answer (PATCH-path normalization) ───────────────────────────

from fastapi import HTTPException

from gradenza_api.routes.quiz import _normalize_mc_answer, _validate_mc_options  # noqa: E402


def test_normalize_mc_answer_lowercase():
    result = _normalize_mc_answer({"question_type": "multiple_choice", "correct_answer": "a"})
    assert result["correct_answer"] == "A"


def test_normalize_mc_answer_with_dot():
    result = _normalize_mc_answer({"question_type": "multiple_choice", "correct_answer": "B."})
    assert result["correct_answer"] == "B"


def test_normalize_mc_answer_already_uppercase():
    result = _normalize_mc_answer({"question_type": "multiple_choice", "correct_answer": "C"})
    assert result["correct_answer"] == "C"


def test_normalize_mc_answer_noop_for_non_mc():
    result = _normalize_mc_answer({"question_type": "short_answer", "correct_answer": "photosynthesis"})
    assert result["correct_answer"] == "photosynthesis"


def test_normalize_mc_answer_noop_for_multi_char():
    # Free-text MCQ answers that aren't a single letter pass through unchanged
    result = _normalize_mc_answer({"question_type": "multiple_choice", "correct_answer": "AB"})
    assert result["correct_answer"] == "AB"


def test_normalize_mc_answer_does_not_mutate_input():
    original = {"question_type": "multiple_choice", "correct_answer": "a"}
    _normalize_mc_answer(original)
    assert original["correct_answer"] == "a"


# ── _validate_mc_options ──────────────────────────────────────────────────────

def _mc_qj(**overrides) -> dict:
    q = {
        "question_type": "multiple_choice",
        "options": ["A. Paris", "B. London", "C. Berlin", "D. Madrid"],
        "correct_answer": "A",
    }
    q.update(overrides)
    return q


def test_validate_mc_options_valid():
    _validate_mc_options(_mc_qj())  # must not raise


def test_validate_mc_options_noop_non_mc():
    # Non-MCQ questions are always accepted regardless of options shape
    _validate_mc_options({"question_type": "short_answer", "options": None, "correct_answer": "photosynthesis"})


def test_validate_mc_options_noop_when_options_none():
    # options=None means the field isn't being updated; skip validation
    _validate_mc_options(_mc_qj(options=None))


def test_validate_mc_options_multi_char_answer_rejected():
    # Core regression: "AB" must be rejected, not silently passed through
    with pytest.raises(HTTPException, match="single letter"):
        _validate_mc_options(_mc_qj(correct_answer="AB"))


def test_validate_mc_options_empty_answer_rejected():
    with pytest.raises(HTTPException, match="single letter"):
        _validate_mc_options(_mc_qj(correct_answer=""))


def test_validate_mc_options_answer_not_in_options_rejected():
    with pytest.raises(HTTPException, match="single letter"):
        _validate_mc_options(_mc_qj(correct_answer="Z"))


def test_validate_mc_options_lowercase_answer_accepted():
    # Lowercase is normalised to upper before comparison
    _validate_mc_options(_mc_qj(correct_answer="a"))


def test_validate_mc_options_answer_with_trailing_dot_accepted():
    _validate_mc_options(_mc_qj(correct_answer="A."))


def test_validate_mc_options_too_few_options_rejected():
    with pytest.raises(HTTPException, match="at least 2"):
        _validate_mc_options(_mc_qj(options=["A. Only"]))


def test_validate_mc_options_empty_list_rejected():
    with pytest.raises(HTTPException, match="at least 2"):
        _validate_mc_options(_mc_qj(options=[]))


def test_validate_mc_options_duplicate_label_rejected():
    with pytest.raises(HTTPException, match="Duplicate"):
        _validate_mc_options(_mc_qj(options=["A. One", "A. Two", "B. Three"]))


def test_validate_mc_options_missing_prefix_rejected():
    with pytest.raises(HTTPException, match="must start with a letter and dot"):
        _validate_mc_options(_mc_qj(options=["Paris", "B. London", "C. Berlin"]))


def test_validate_mc_options_non_string_option_rejected():
    with pytest.raises(HTTPException, match="must be a string"):
        _validate_mc_options(_mc_qj(options=["A. Paris", 42]))


def test_validate_mc_options_uploaded_image_slot_ignored():
    # image_slot as a dict (uploaded image) must not interfere with option validation
    _validate_mc_options(_mc_qj(image_slot={"url": "https://example.com/img.png", "position": "above"}))
