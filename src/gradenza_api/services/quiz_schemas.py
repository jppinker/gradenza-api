"""Pydantic schema for LLM-generated quiz JSON."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_OPTION_LETTER_RE = re.compile(r"^\s*([A-Za-z])\.")


class QuizQuestionSchema(BaseModel):
    slot: int = Field(..., ge=1)
    question_type: Literal["multiple_choice", "fill_blank", "short_answer", "long_answer"]
    difficulty: Literal["easy", "medium", "challenge"]
    bloom_level: Literal["remember", "understand", "apply", "analyse", "evaluate", "create"]
    question_text: str = Field(..., min_length=1)
    options: list[str] | None = None
    correct_answer: str = Field(..., min_length=1)
    acceptable_answers: list[str] | None = None
    explanation: str = Field(..., min_length=1)
    marks: int = Field(..., ge=1)
    rubric: str | None = None
    answer_space: Literal["multiple_choice", "single_line", "short_paragraph", "long_paragraph"]
    source_reference: str | None = None
    # LLM always emits null; uploaded images are stored on the DB row directly.
    image_slot: None = None

    @model_validator(mode="after")
    def validate_mc_shape(self) -> "QuizQuestionSchema":
        if self.question_type != "multiple_choice":
            return self

        opts = self.options or []
        if len(opts) < 2:
            raise ValueError("multiple_choice questions must have at least 2 options")

        option_letters: set[str] = set()
        for opt in opts:
            m = _OPTION_LETTER_RE.match(opt)
            if not m:
                raise ValueError(f"option {opt!r} must start with a letter and dot, e.g. 'A. text'")
            letter = m.group(1).upper()
            if letter in option_letters:
                raise ValueError(f"duplicate option label {letter!r}")
            option_letters.add(letter)

        # Normalize and store the canonical uppercase letter.
        answer_letter = self.correct_answer.strip().upper().rstrip(".")
        if len(answer_letter) != 1 or answer_letter not in option_letters:
            raise ValueError(
                f"correct_answer {self.correct_answer!r} must be a single letter matching one of the options"
            )
        self.correct_answer = answer_letter

        return self

    model_config = {"extra": "allow"}


class QuizBlueprintItemSchema(BaseModel):
    slot: int = Field(..., ge=1)
    question_type: Literal["multiple_choice", "fill_blank", "short_answer", "long_answer"]
    difficulty: Literal["easy", "medium", "challenge"]
    marks: int = Field(..., ge=1, le=5)

    model_config = {"extra": "forbid"}


def validate_quiz_blueprint_items(items: Any, *, expected_count: int) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError(f"Expected blueprint JSON array, got {type(items).__name__}")
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} blueprint items, got {len(items)}")

    validated = [QuizBlueprintItemSchema.model_validate(it).model_dump() for it in items]

    slots = [it["slot"] for it in validated]
    if len(set(slots)) != len(slots):
        raise ValueError("Blueprint slot values must be unique")
    if sorted(slots) != list(range(1, expected_count + 1)):
        raise ValueError(f"Blueprint slot values must be 1..{expected_count}")

    return sorted(validated, key=lambda x: x["slot"])
