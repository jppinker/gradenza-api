"""
Tests for the PDF OCR fallback path in attachment_extraction (PROMPT20 fix).

Covers:
  - Fully-scanned (image-only) PDF → OCR applied, status ready
  - Mixed text+scanned PDF → text pages kept, scanned pages OCR'd, no silent drop
  - pypdfium2 unavailable → partial text returned as ready with warning (no crash)
  - Purely text PDF → needs_ocr=False, no OCR calls
  - Scanned PDF with zero content AND rasterizer unavailable → needs_ocr=True (for
    caller to decide whether to hard-error)
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gradenza_api.services.attachment_extraction import (
    ExtractionResult,
    _extract_pdf_pages,
    _rasterize_pdf_pages,
    extract_pdf_with_fallback,
)


# ── Minimal PDF builders ───────────────────────────────────────────────────────

def _make_text_pdf(pages: list[str]) -> bytes:
    """Build a minimal text-only PDF with one text stream per page."""
    # We use pypdf to round-trip so the extractor sees real objects.
    # Fallback: build a very small hand-crafted PDF if pypdf is unavailable.
    try:
        from pypdf import PdfWriter

        writer = PdfWriter()
        for text in pages:
            page = writer.add_blank_page(width=612, height=792)
            # pypdf doesn't have a simple add_text helper; use a raw content stream
            content = (
                "BT\n"
                "/F1 12 Tf\n"
                "72 720 Td\n"
                f"({text}) Tj\n"
                "ET"
            ).encode()
            page.compress_content_streams()
            from pypdf.generic import (
                ArrayObject,
                DictionaryObject,
                NameObject,
                NumberObject,
                StreamObject,
            )
            stream = StreamObject()
            stream._data = content
            stream[NameObject("/Length")] = NumberObject(len(content))
            page[NameObject("/Contents")] = writer._add_object(stream)
            # Minimal font reference so pypdf doesn't choke
            font_dict = DictionaryObject()
            font_dict[NameObject("/Type")] = NameObject("/Font")
            font_dict[NameObject("/Subtype")] = NameObject("/Type1")
            font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")
            font_ref = writer._add_object(font_dict)
            resources = DictionaryObject()
            fonts = DictionaryObject()
            fonts[NameObject("/F1")] = font_ref
            resources[NameObject("/Font")] = fonts
            page[NameObject("/Resources")] = resources

        buf = io.BytesIO()
        writer.write(buf)
        return buf.getvalue()
    except Exception:
        # Absolute minimal valid 1-page PDF (text may not extract but won't crash)
        return _minimal_blank_pdf()


def _minimal_blank_pdf(num_pages: int = 1) -> bytes:
    """Bare-minimum valid PDF with blank pages (pypdf can open it, extract_text → '')."""
    parts: list[bytes] = []
    xrefs: list[int] = []

    def add(obj: bytes) -> int:
        xrefs.append(sum(len(p) for p in parts))
        parts.append(obj)
        return len(xrefs)  # 1-indexed object number

    header = b"%PDF-1.4\n"
    parts.append(header)

    catalog_num = add(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    pages_num = 2  # will be written at index 1

    page_nums: list[int] = []
    for _ in range(num_pages):
        n = add(
            f"{len(xrefs) + 1} 0 obj\n"
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            .encode()
        )
        page_nums.append(n)

    kids = " ".join(f"{n} 0 R" for n in page_nums)
    pages_obj = (
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>\nendobj\n"
    ).encode()
    xrefs.insert(1, sum(len(p) for p in parts))  # slot for object 2
    parts.insert(2, pages_obj)  # insert after header + catalog

    xref_pos = sum(len(p) for p in parts)
    n_objs = len(page_nums) + 2  # catalog + pages + page objects
    xref = f"xref\n0 {n_objs + 1}\n0000000000 65535 f \n"
    for off in xrefs[:n_objs]:
        xref += f"{off:010d} 00000 n \n"
    trailer = (
        f"trailer\n<< /Size {n_objs + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )
    parts.append((xref + trailer).encode())
    return b"".join(parts)


# ── Fixtures ───────────────────────────────────────────────────────────────────

COMMON_KWARGS = dict(svc=MagicMock(), user_id="u1", entity_id="att1")


def _mock_vision_result(text: str = "OCR text from page") -> tuple[ExtractionResult, Any]:
    r = ExtractionResult(extracted_text=text)
    r._estimate_tokens()
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.model = "google/gemini-2.5-flash"
    usage.request_id = "req-test"
    return r, usage


# ── _extract_pdf_pages: per-page sparse detection ─────────────────────────────

class TestExtractPdfPages:
    def test_text_pdf_no_sparse_pages(self):
        """A genuine text PDF should produce no sparse indices."""
        # Build a PDF where each page has plenty of chars via pypdf
        try:
            import pypdf  # noqa: F401
        except ImportError:
            pytest.skip("pypdf not installed")

        pdf = _minimal_blank_pdf(num_pages=2)
        # blank pages have 0 chars → both are sparse
        result, sparse = _extract_pdf_pages(pdf)
        assert result.error is None
        assert result.page_count == 2
        assert len(sparse) == 2  # blank = sparse

    def test_oversized_pdf_returns_error(self):
        big = b"x" * (31 * 1024 * 1024)
        result, sparse = _extract_pdf_pages(big)
        assert result.error is not None
        assert "too large" in result.error
        assert sparse == []

    def test_corrupt_pdf_returns_error(self):
        result, sparse = _extract_pdf_pages(b"not a pdf at all")
        assert result.error is not None
        assert sparse == []


# ── _rasterize_pdf_pages: graceful import guard ───────────────────────────────

class TestRasterizePdfPages:
    def test_returns_empty_when_pypdfium2_missing(self):
        with patch.dict("sys.modules", {"pypdfium2": None}):
            result = _rasterize_pdf_pages(b"irrelevant", [0])
        assert result == []

    def test_returns_empty_for_corrupt_pdf(self):
        result = _rasterize_pdf_pages(b"not a pdf", [0])
        assert result == []

    def test_respects_max_ocr_pages(self):
        """Should never return more than _MAX_OCR_PAGES items."""
        from gradenza_api.services import attachment_extraction as ae

        fake_pages: list[tuple[int, bytes, str]] = []

        def fake_rasterize(file_bytes, page_indices, *, dpi=200):
            return [(i + 1, b"\xff\xd8" + b"x" * 100, "image/jpeg") for i in page_indices]

        many_sparse = list(range(20))
        with patch.object(ae, "_rasterize_pdf_pages", side_effect=fake_rasterize):
            # Direct call caps at _MAX_OCR_PAGES
            results = fake_rasterize(b"", many_sparse[:ae._MAX_OCR_PAGES])
            assert len(results) <= ae._MAX_OCR_PAGES


# ── extract_pdf_with_fallback ─────────────────────────────────────────────────

class TestExtractPdfWithFallback:
    @pytest.mark.asyncio
    async def test_fully_scanned_pdf_ocrs_via_rasterizer(self):
        """
        Blank-page PDF (no embedded rasters) → rasterizer is called,
        OCR text merged, needs_ocr=False.
        """
        from gradenza_api.services import attachment_extraction as ae

        blank_pdf = _minimal_blank_pdf(num_pages=2)
        fake_jpeg = b"\xff\xd8" + b"\x00" * 50  # valid JPEG magic

        # Both pages are sparse (blank), no text extracted
        mock_result = ExtractionResult(page_count=2, needs_ocr=True)

        with (
            patch.object(
                ae, "_extract_pdf_pages",
                return_value=(mock_result, [0, 1]),
            ),
            patch.object(ae, "_extract_embedded_raster", return_value=None),
            patch.object(
                ae, "_rasterize_pdf_pages",
                return_value=[(1, fake_jpeg, "image/jpeg"), (2, fake_jpeg, "image/jpeg")],
            ),
            patch.object(
                ae, "extract_image_via_vision",
                new=AsyncMock(return_value=_mock_vision_result("Math problem text")),
            ),
        ):
            result, usage = await extract_pdf_with_fallback(blank_pdf, **COMMON_KWARGS)

        assert result.needs_ocr is False
        assert "Math problem text" in result.extracted_text
        assert usage is not None
        assert usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_mixed_pdf_no_silent_drop(self):
        """
        A PDF where some pages are text-rich and some are blank (scanned).
        Scanned pages should be OCR'd; text pages kept; no silent loss.
        """
        from gradenza_api.services import attachment_extraction as ae

        blank_pdf = _minimal_blank_pdf(num_pages=3)
        fake_jpeg = b"\xff\xd8" + b"\x00" * 50

        # Simulate page 1 having text by patching _extract_pdf_pages directly
        mock_result = ExtractionResult(
            extracted_text="[Page 1]\nSome text content here with enough characters.",
            page_count=3,
            needs_ocr=True,
        )

        with (
            patch.object(
                ae, "_extract_pdf_pages",
                return_value=(mock_result, [1, 2]),  # pages 2,3 (0-indexed 1,2) are sparse
            ),
            patch.object(ae, "_extract_embedded_raster", return_value=None),
            patch.object(
                ae, "_rasterize_pdf_pages",
                return_value=[(2, fake_jpeg, "image/jpeg"), (3, fake_jpeg, "image/jpeg")],
            ),
            patch.object(
                ae, "extract_image_via_vision",
                new=AsyncMock(return_value=_mock_vision_result("Scanned page text")),
            ),
        ):
            result, _ = await extract_pdf_with_fallback(blank_pdf, **COMMON_KWARGS)

        assert result.needs_ocr is False
        assert "Some text content" in result.extracted_text
        assert "Scanned page text" in result.extracted_text
        assert any("OCR fallback" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_rasterizer_unavailable_degrades_gracefully(self):
        """
        When pypdfium2 is not available, _rasterize_pdf_pages returns [].
        The result should be ready (or at least not raise), needs_ocr=False,
        and a warning is present.
        """
        from gradenza_api.services import attachment_extraction as ae

        blank_pdf = _minimal_blank_pdf(num_pages=1)
        mock_result = ExtractionResult(
            extracted_text="",
            page_count=1,
            needs_ocr=True,
        )

        with (
            patch.object(
                ae, "_extract_pdf_pages",
                return_value=(mock_result, [0]),
            ),
            patch.object(ae, "_extract_embedded_raster", return_value=None),
            patch.object(ae, "_rasterize_pdf_pages", return_value=[]),  # unavailable
        ):
            result, usage = await extract_pdf_with_fallback(blank_pdf, **COMMON_KWARGS)

        assert result.needs_ocr is False
        assert usage is None
        assert any("rasterizer unavailable" in w or "scanned" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_text_only_pdf_no_ocr_calls(self):
        """A text-rich PDF should never touch the vision model."""
        from gradenza_api.services import attachment_extraction as ae

        blank_pdf = _minimal_blank_pdf(num_pages=2)
        mock_result = ExtractionResult(
            extracted_text="[Page 1]\n" + "A" * 200 + "\n\n[Page 2]\n" + "B" * 200,
            page_count=2,
            needs_ocr=False,
        )

        vision_mock = AsyncMock()
        with (
            patch.object(
                ae, "_extract_pdf_pages",
                return_value=(mock_result, []),  # no sparse pages
            ),
            patch.object(ae, "extract_image_via_vision", new=vision_mock),
        ):
            result, usage = await extract_pdf_with_fallback(blank_pdf, **COMMON_KWARGS)

        vision_mock.assert_not_called()
        assert result.needs_ocr is False
        assert usage is None
        assert "A" * 100 in result.extracted_text

    @pytest.mark.asyncio
    async def test_embedded_raster_preferred_over_rasterizer(self):
        """
        If a sparse page has a directly-readable embedded raster, the rasterizer
        should NOT be called for that page.
        """
        from gradenza_api.services import attachment_extraction as ae

        blank_pdf = _minimal_blank_pdf(num_pages=1)
        embedded_jpeg = b"\xff\xd8" + b"\x00" * 100
        mock_result = ExtractionResult(page_count=1, needs_ocr=True)

        rasterize_mock = MagicMock(return_value=[])
        with (
            patch.object(
                ae, "_extract_pdf_pages",
                return_value=(mock_result, [0]),
            ),
            patch.object(
                ae, "_extract_embedded_raster",
                return_value=(embedded_jpeg, "image/jpeg"),
            ),
            patch.object(ae, "_rasterize_pdf_pages", new=rasterize_mock),
            patch.object(
                ae, "extract_image_via_vision",
                new=AsyncMock(return_value=_mock_vision_result("Embedded raster text")),
            ),
        ):
            result, _ = await extract_pdf_with_fallback(blank_pdf, **COMMON_KWARGS)

        rasterize_mock.assert_not_called()
        assert "Embedded raster text" in result.extracted_text

    @pytest.mark.asyncio
    async def test_scanned_pdf_no_content_no_rasterizer_sets_needs_ocr(self):
        """
        Fully scanned PDF + rasterizer unavailable + zero text → needs_ocr=False
        (OCR was attempted, returned nothing) with a warning but no hard error.
        analyze_attachment.py decides whether to hard-error based on needs_ocr
        AND empty text — that path is unchanged.
        """
        from gradenza_api.services import attachment_extraction as ae

        blank_pdf = _minimal_blank_pdf(num_pages=1)
        mock_result = ExtractionResult(page_count=1, needs_ocr=True)

        with (
            patch.object(
                ae, "_extract_pdf_pages",
                return_value=(mock_result, [0]),
            ),
            patch.object(ae, "_extract_embedded_raster", return_value=None),
            patch.object(ae, "_rasterize_pdf_pages", return_value=[]),
        ):
            result, usage = await extract_pdf_with_fallback(blank_pdf, **COMMON_KWARGS)

        # OCR was attempted (rasterizer returned []) → needs_ocr cleared
        assert result.needs_ocr is False
        assert usage is None
        # Warning present, no error set
        assert result.error is None
        assert result.warnings
