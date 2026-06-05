"""
Tests for attachment_context.build_attachment_context and _redact_pii (PROMPT28).
"""

import pytest
from gradenza_api.services.attachment_context import _redact_pii, build_attachment_context


# ── _redact_pii ───────────────────────────────────────────────────────────────

class TestRedactPii:
    def test_redacts_email(self):
        text, n = _redact_pii("Contact john.doe@school.example.com for details.")
        assert "john.doe@school.example.com" not in text
        assert "[email redacted]" in text
        assert n == 1

    def test_redacts_multiple_emails(self):
        text, n = _redact_pii("From alice@x.com to bob@y.org.")
        assert n == 2
        assert "[email redacted]" in text

    def test_redacts_us_ssn_dash(self):
        text, n = _redact_pii("SSN: 123-45-6789.")
        assert "123-45-6789" not in text
        assert "[ID redacted]" in text
        assert n == 1

    def test_redacts_us_ssn_space(self):
        text, n = _redact_pii("SSN 123 45 6789")
        assert "123 45 6789" not in text
        assert n == 1

    def test_redacts_uk_ni_number(self):
        text, n = _redact_pii("NI: AB123456C")
        assert "AB123456C" not in text
        assert "[ID redacted]" in text
        assert n >= 1

    def test_redacts_nanp_phone_dashes(self):
        text, n = _redact_pii("Call 555-867-5309.")
        assert "555-867-5309" not in text
        assert "[phone redacted]" in text
        assert n == 1

    def test_redacts_nanp_phone_parens(self):
        text, n = _redact_pii("Phone: (416) 555-1234")
        assert "(416) 555-1234" not in text
        assert n == 1

    def test_redacts_international_phone(self):
        text, n = _redact_pii("Call +44 7911 123456 now.")
        assert "+44 7911 123456" not in text
        assert "[phone redacted]" in text
        assert n == 1

    def test_no_false_positive_on_math(self):
        # Pure digit sequences in math context must not be redacted
        text, n = _redact_pii("Solve x^2 + 3x + 2 = 0. Answer: x = 1234567890.")
        assert n == 0

    def test_no_false_positive_on_year_range(self):
        text, n = _redact_pii("Between 1900 and 2024.")
        assert n == 0

    def test_no_redaction_needed(self):
        clean = "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a."
        text, n = _redact_pii(clean)
        assert text == clean
        assert n == 0


# ── build_attachment_context ──────────────────────────────────────────────────

def _att(**kwargs):
    defaults = dict(id="att-1", file_name="test.pdf", file_type="pdf",
                    summary="", extracted_text="", analysis_json={})
    return {**defaults, **kwargs}


class TestBuildAttachmentContextPii:
    def test_pii_in_extracted_text_is_redacted(self):
        att = _att(extracted_text="Student email: alice@school.edu. Score: 95.")
        ctx, meta, warnings = build_attachment_context([att])
        assert "alice@school.edu" not in ctx
        assert "[email redacted]" in ctx
        assert any("redacted" in w for w in warnings)

    def test_warning_includes_filename_and_count(self):
        att = _att(
            file_name="grade_sheet.pdf",
            extracted_text="a@b.com c@d.com",
        )
        _, _, warnings = build_attachment_context([att])
        pii_warnings = [w for w in warnings if "redacted" in w]
        assert len(pii_warnings) == 1
        assert "grade_sheet.pdf" in pii_warnings[0]
        assert "2" in pii_warnings[0]

    def test_summary_not_redacted(self):
        # Summaries are AI-generated; we trust them and don't touch them.
        att = _att(summary="Teacher alice@school.edu uploaded this.", extracted_text="")
        ctx, _, warnings = build_attachment_context([att])
        assert "alice@school.edu" in ctx
        assert not any("redacted" in w for w in warnings)

    def test_clean_text_produces_no_pii_warning(self):
        att = _att(extracted_text="Integrate f(x) = x^2 from 0 to 1.")
        _, _, warnings = build_attachment_context([att])
        assert not any("redacted" in w for w in warnings)

    def test_empty_attachments(self):
        ctx, meta, warnings = build_attachment_context([])
        assert ctx == ""
        assert meta == []
        assert warnings == []
