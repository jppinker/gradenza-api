"""
Shared attachment context builder (PROMPT16).

Converts online_lesson_attachments data into a token-budgeted prompt
context string for injection into AI system prompts.

Used by: chat, generate-plan, generate-questions, generate-homework,
         and question recommendations — so all AI endpoints share the
         same truncation / deduplication / labelling logic.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# Default character budgets
DEFAULT_MAX_TOTAL_CHARS: int = 8_000
DEFAULT_MAX_EXTRACTED_PER_ATTACHMENT: int = 2_000


# ── PII redaction (PROMPT28) ──────────────────────────────────────────────────

# Email addresses — low false-positive risk in educational text.
_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.ASCII,
)

# UK National Insurance numbers: two letters, six digits, optional A-D suffix.
_RE_UK_NI = re.compile(
    r"\b(?!BG|GB|NK|KN|NT|TN|ZZ)[A-CEGHJ-PR-TW-Z]{2}\d{6}[ABCD]?\b",
    re.IGNORECASE,
)

# US Social Security Numbers: 123-45-6789 or 123 45 6789.
_RE_US_SSN = re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b")

# Phone numbers — conservative patterns only to avoid false positives on maths:
#   North American NANP:  (123) 456-7890 / 123-456-7890 / 123.456.7890
#   Explicit international: +44 7911 123456 / +1-800-555-0100
_RE_PHONE_NANP = re.compile(
    r"\b\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b"
)
_RE_PHONE_INTL = re.compile(
    r"\+\d{1,3}(?:[\s\-]\d{2,5}){1,3}[\s\-]\d{3,6}\b"
)

_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_RE_EMAIL,      "[email redacted]"),
    (_RE_UK_NI,      "[ID redacted]"),
    (_RE_US_SSN,     "[ID redacted]"),
    (_RE_PHONE_NANP, "[phone redacted]"),
    (_RE_PHONE_INTL, "[phone redacted]"),
]


def _redact_pii(text: str) -> tuple[str, int]:
    """
    Replace obvious PII patterns with placeholder tokens.

    Returns the redacted text and the total number of substitutions made.
    Applies only to extracted (raw) text — AI-generated summaries are
    already processed and unlikely to contain raw PII.
    """
    total = 0
    for pattern, placeholder in _PII_PATTERNS:
        text, n = pattern.subn(placeholder, text)
        total += n
    return text, total


# ── Context builder ───────────────────────────────────────────────────────────

def build_attachment_context(
    attachments: list[dict[str, Any]],
    *,
    max_budget: int = DEFAULT_MAX_TOTAL_CHARS,
    header: str = "## Attached materials",
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """
    Build a token-budgeted attachment context block for an AI system prompt.

    Parameters
    ----------
    attachments:
        List of attachment payload dicts. Expected keys (all optional with safe
        defaults): id, file_name, file_type, summary, extracted_text, analysis_json.
        ``extracted_text`` may be nested inside ``analysis_json`` as a fallback.
    max_budget:
        Maximum total characters of attachment content included in the output.
    header:
        Markdown section heading prepended to the block.

    Returns
    -------
    context_str:
        Formatted Markdown block ready to append to a system prompt. Empty
        string when there are no attachments or all budget is exhausted before
        any content is added.
    metadata:
        Lightweight dicts ``{id, file_name, file_type}`` for every attachment
        processed (including omitted ones), suitable for source_meta recording.
    warnings:
        Human-readable warnings (e.g. budget exceeded, duplicates skipped,
        PII detected and redacted).
    """
    if not attachments:
        return "", [], []

    metadata: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_hashes: set[str] = set()  # content-based deduplication
    total_chars: int = 0
    count: int = len(attachments)
    parts: list[str] = [f"{header} ({count} file{'s' if count != 1 else ''})"]

    for i, att in enumerate(attachments):
        file_name = str(att.get("file_name") or "file")
        file_type = str(att.get("file_type") or "unknown")
        att_id = str(att.get("id") or "")
        summary = str(att.get("summary") or "").strip()

        # Resolved extracted_text: top-level key wins, analysis_json excerpt as fallback
        extracted_text = str(att.get("extracted_text") or "").strip()
        if not extracted_text:
            aj = att.get("analysis_json") or {}
            if isinstance(aj, dict):
                extracted_text = str(aj.get("extracted_text") or "").strip()

        # ── PII redaction (PROMPT28) ───────────────────────────────────────────
        # Applied to both raw extracted text and AI-generated summaries, since
        # summaries may echo student names or IDs from the source document.
        if summary:
            summary, pii_count_s = _redact_pii(summary)
            if pii_count_s:
                warnings.append(
                    f"{pii_count_s} potential personal data item(s) redacted from {file_name!r} summary before sending to AI."
                )
        if extracted_text:
            extracted_text, pii_count = _redact_pii(extracted_text)
            if pii_count:
                warnings.append(
                    f"{pii_count} potential personal data item(s) redacted from {file_name!r} before sending to AI."
                )

        # Page / slide counts from analysis_json (secondary source after top-level)
        aj = att.get("analysis_json") or {}
        page_count: int | None = att.get("page_count") or (
            int(aj["page_count"]) if isinstance(aj, dict) and aj.get("page_count") else None
        )
        slide_count: int | None = att.get("slide_count") or (
            int(aj["slide_count"]) if isinstance(aj, dict) and aj.get("slide_count") else None
        )

        metadata.append({"id": att_id, "file_name": file_name, "file_type": file_type})

        # Budget exhausted — note remaining attachments and stop
        if total_chars >= max_budget:
            remaining = count - i
            msg = f"{remaining} attachment(s) omitted — token budget reached."
            warnings.append(msg)
            parts.append(f"[{remaining} more attachment(s) omitted — token budget reached]")
            break

        label = f"### [{i + 1}] {file_name} ({file_type})"
        parts.append(label)

        # ── Summary (always included first; short and high-value) ──────────────
        if summary:
            summary_hash = hashlib.md5(summary.encode()).hexdigest()
            if summary_hash in seen_hashes:
                warnings.append(f"Duplicate summary for {file_name!r} skipped.")
            else:
                snippet = summary[:500]
                parts.append(f"Summary: {snippet}")
                total_chars += len(snippet)
                seen_hashes.add(summary_hash)

        # ── Page / slide label ─────────────────────────────────────────────────
        if page_count:
            parts.append(f"Pages: {page_count}")
        elif slide_count:
            parts.append(f"Slides: {slide_count}")

        # ── Extracted text excerpt up to per-attachment + remaining budget ─────
        if extracted_text:
            text_fingerprint = hashlib.md5(extracted_text[:200].encode()).hexdigest()
            if text_fingerprint in seen_hashes:
                warnings.append(f"Duplicate content for {file_name!r} skipped.")
            else:
                budget_left = max_budget - total_chars
                per_att = min(DEFAULT_MAX_EXTRACTED_PER_ATTACHMENT, budget_left)
                if per_att > 100:
                    excerpt = extracted_text[:per_att]
                    if len(extracted_text) > per_att:
                        excerpt += f"\n[… {len(extracted_text) - per_att:,} chars omitted]"
                    parts.append(f"Content:\n{excerpt}")
                    total_chars += len(excerpt)
                    seen_hashes.add(text_fingerprint)
                elif budget_left <= 100:
                    warnings.append(
                        f"Token budget exhausted — extracted text for {file_name!r} omitted."
                    )

    context_str = "\n\n".join(parts) if len(parts) > 1 else ""
    return context_str, metadata, warnings
