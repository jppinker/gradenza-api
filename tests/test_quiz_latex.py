"""Tests for LaTeX/KaTeX support in quiz generation and export.

No database, network, or environment variables required.
"""
import html as _html

import pytest

from gradenza_api.jobs.generate_quiz import _GENERATE_PROMPT
from gradenza_api.routes.quiz import (
    _KATEX_HEAD,
    _REGENERATE_PROMPT,
    _generate_quiz_html,
    _h,
)


# ── Prompt instructions ──────────────────────────────────────────────────────


def test_generate_prompt_has_inline_delimiter_instruction():
    assert r"\( ... \)" in _GENERATE_PROMPT or "\\\\(" in _GENERATE_PROMPT or "\\(" in _GENERATE_PROMPT


def test_generate_prompt_has_display_delimiter_instruction():
    assert r"\[ ... \]" in _GENERATE_PROMPT or "\\\\[" in _GENERATE_PROMPT or "\\[" in _GENERATE_PROMPT


def test_generate_prompt_covers_all_fields():
    """Prompt must mention all output fields that can contain math."""
    lower = _GENERATE_PROMPT.lower()
    assert "question_text" in lower
    assert "explanation" in lower
    assert "options" in lower


def test_regenerate_prompt_has_latex_instructions():
    assert "\\(" in _REGENERATE_PROMPT or "\\\\(" in _REGENERATE_PROMPT


def test_regenerate_prompt_has_display_delimiter():
    assert "\\[" in _REGENERATE_PROMPT or "\\\\[" in _REGENERATE_PROMPT


# ── HTML escaping preserves LaTeX delimiters ─────────────────────────────────


def test_h_preserves_backslash():
    assert _h("\\(") == "\\("


def test_h_preserves_latex_inline():
    result = _h(r"\( x^2 + 3x \)")
    assert result == r"\( x^2 + 3x \)"


def test_h_preserves_latex_display():
    result = _h(r"\[ E = mc^2 \]")
    assert result == r"\[ E = mc^2 \]"


def test_h_escapes_html_special_chars():
    """< > & are still escaped so they round-trip safely through the browser DOM."""
    result = _h("x < y & z > 0")
    assert "&lt;" in result
    assert "&gt;" in result
    assert "&amp;" in result


def test_h_latex_with_lt_gt():
    """LaTeX containing < > is HTML-escaped but the browser decodes it before KaTeX sees it."""
    result = _h(r"\( x < y \)")
    assert "\\(" in result
    assert "&lt;" in result


# ── KaTeX head constant ──────────────────────────────────────────────────────


def test_katex_head_has_css_link():
    assert "katex.min.css" in _KATEX_HEAD


def test_katex_head_has_katex_js():
    assert "katex.min.js" in _KATEX_HEAD


def test_katex_head_has_auto_render_js():
    assert "auto-render.min.js" in _KATEX_HEAD


def test_katex_head_has_render_math_in_element_call():
    assert "renderMathInElement" in _KATEX_HEAD


def test_katex_head_sets_pdf_ready_flag():
    assert "window.__PDF_READY" in _KATEX_HEAD


def test_katex_head_configures_backslash_paren_delimiters():
    # The JS string in _KATEX_HEAD must contain \\( and \\) as JS string literals
    # which in the Python source are written as \\\\( (4 backslashes → 2 in string → JS \\()
    assert "\\\\(" in _KATEX_HEAD


def test_katex_head_configures_backslash_bracket_delimiters():
    assert "\\\\[" in _KATEX_HEAD


# ── Generated HTML contains KaTeX ────────────────────────────────────────────


def _make_session(**overrides: object) -> dict:
    base: dict = {
        "id": "sess-001",
        "title": "Physics Quiz",
        "subject": "Physics",
        "grade_level": "IB Year 1",
    }
    base.update(overrides)
    return base


def _make_question(question_text: str, **overrides: object) -> dict:
    qj: dict = {
        "slot": 1,
        "question_type": "short_answer",
        "difficulty": "medium",
        "question_text": question_text,
        "options": None,
        "correct_answer": r"\( v = d/t \)",
        "explanation": r"Speed equals distance divided by time.",
        "marks": 2,
        "rubric": None,
        "answer_space": "short_paragraph",
        "image_slot": None,
    }
    qj.update(overrides)
    return {"id": "q-001", "slot": 1, "question_json": qj}


def test_html_includes_katex_css():
    session = _make_session()
    questions = [_make_question(r"What is \( v \)?")]
    html = _generate_quiz_html(session, questions, "exam_style", {})
    assert "katex.min.css" in html


def test_html_includes_katex_js():
    session = _make_session()
    questions = [_make_question(r"What is \( v \)?")]
    html = _generate_quiz_html(session, questions, "exam_style", {})
    assert "katex.min.js" in html


def test_html_includes_pdf_ready_flag():
    session = _make_session()
    questions = [_make_question(r"What is \( v \)?")]
    html = _generate_quiz_html(session, questions, "exam_style", {})
    assert "window.__PDF_READY" in html


def test_html_preserves_inline_math_in_question_text():
    session = _make_session()
    questions = [_make_question(r"The velocity is \( v = d/t \).")]
    html = _generate_quiz_html(session, questions, "exam_style", {})
    # backslash is not HTML-special so \( survives _h() and appears in the HTML
    assert r"\( v = d/t \)" in html


def test_html_preserves_display_math():
    session = _make_session()
    questions = [_make_question(r"Show that \[ E = mc^2 \] is correct.")]
    html = _generate_quiz_html(session, questions, "exam_style", {})
    assert r"\[ E = mc^2 \]" in html


def test_teacher_html_shows_answer_with_math():
    session = _make_session()
    questions = [_make_question(r"Find \( F \).", correct_answer=r"\( F = ma \)")]
    html = _generate_quiz_html(session, questions, "exam_style", {}, version="teacher")
    assert r"\( F = ma \)" in html


def test_teacher_html_shows_rubric_with_math():
    session = _make_session()
    questions = [_make_question(
        r"Derive \( E = mc^2 \).",
        rubric=r"Award 1 mark for \( E = mc^2 \), 1 mark for clear derivation.",
        correct_answer="See rubric.",
    )]
    html = _generate_quiz_html(session, questions, "exam_style", {}, version="teacher")
    assert "Rubric:" in html
    assert r"\( E = mc^2 \)" in html


def test_html_no_script_injection_in_question_text():
    """Malicious HTML in question_text must be escaped, not injected."""
    session = _make_session()
    questions = [_make_question('<script>alert(1)</script>')]
    html = _generate_quiz_html(session, questions, "exam_style", {})
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
