"""
Playwright-based HTML → PDF rendering.

We use Playwright (Chromium) instead of WeasyPrint because Class PDF Notes rely on:
- KaTeX/JS rendering for LaTeX math
- Remote images (Supabase public URLs)
- Canvas-rendered graphs
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CHROMIUM_ARGS: list[str] = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-dev-shm-usage",
]


def get_playwright_chromium_executable_path() -> str | None:
    """
    Best-effort lookup of the Chromium executable Playwright expects to use.

    This does not launch Chromium; it only asks Playwright where it *expects*
    the bundled executable to be located, then callers can check file existence.
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return None

    try:
        with sync_playwright() as p:
            return p.chromium.executable_path
    except Exception as exc:
        logger.warning("[playwright] unable to determine chromium executable_path: %s", exc)
        return None


def get_playwright_runtime_diagnostics() -> dict[str, str]:
    """
    Return small, log-friendly Playwright diagnostics for production debugging.
    """
    diag: dict[str, str] = {
        "PLAYWRIGHT_BROWSERS_PATH": os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""),
    }

    chromium_path = get_playwright_chromium_executable_path()
    diag["chromium_executable_path"] = chromium_path or ""
    if chromium_path:
        diag["chromium_executable_exists"] = str(Path(chromium_path).is_file())
    return diag


async def render_html_to_pdf_bytes(html: str) -> tuple[bytes, list[str]]:
    """
    Render a full HTML document to PDF bytes using Playwright Chromium.

    Notes:
    - We launch Chromium per-call (simple + stateless). If this becomes hot,
      consider pooling a browser instance.
    - We wait for a custom readiness flag set by the page (`window.__PDF_READY`)
      to ensure math/images/canvas are ready before printing.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "playwright is required for PDF export. "
            "Add 'playwright' to requirements.txt and install browsers "
            "with 'playwright install chromium'."
        ) from exc

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                *_DEFAULT_CHROMIUM_ARGS,
            ]
        )
        try:
            warnings: list[str] = []

            def _add_warning(msg: str) -> None:
                msg = (msg or "").strip()
                if not msg:
                    return
                if msg in warnings:
                    return
                warnings.append(msg)

            page = await browser.new_page()
            await page.set_content(html, wait_until="networkidle")

            # Prefer a deterministic signal from the page. Fallback to a small delay.
            try:
                await page.wait_for_function(
                    "window.__PDF_READY === true",
                    timeout=15_000,
                )
            except Exception:
                logger.warning("[pdf] readiness flag not observed; continuing anyway")
                _add_warning("PDF export readiness signal was not observed; output may be incomplete.")
                await page.wait_for_timeout(750)

            # Pull best-effort warnings from the page (e.g., KaTeX not loaded, broken image placeholders).
            try:
                page_warnings = await page.evaluate(
                    "() => Array.isArray(window.__PDF_WARNINGS) ? window.__PDF_WARNINGS : []"
                )
                if isinstance(page_warnings, list):
                    for w in page_warnings:
                        if isinstance(w, str):
                            _add_warning(w)
            except Exception:
                pass

            # Guardrail: detect broken images, but do not abort the export.
            broken_images: list[str] = await page.evaluate(
                """() => Array.from(document.images || [])
                    .filter(img => img.naturalWidth === 0)
                    .map(img => img.currentSrc || img.src)
                    .filter(Boolean)
                """
            )
            if broken_images:
                logger.warning("[pdf] some images failed to load (%d); sample=%s", len(broken_images), broken_images[:5])
                _add_warning("Some images failed to load and may be missing in the PDF")

            pdf_bytes = await page.pdf(
                format="A4",
                print_background=True,
                margin={
                    "top": "20mm",
                    "right": "16mm",
                    "bottom": "20mm",
                    "left": "16mm",
                },
            )
            return pdf_bytes, warnings
        finally:
            await browser.close()
