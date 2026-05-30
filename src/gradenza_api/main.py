"""
Gradenza API — FastAPI application entry point.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from urllib.parse import urlparse

from arq import create_pool
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gradenza_api.redis_config import build_redis_settings
from gradenza_api.routes import analytics, canvas_chat, class_pdf_notes, materials, ocr, online_lessons, quiz, submissions
from gradenza_api.services.pdf_playwright import get_playwright_runtime_diagnostics
from gradenza_api.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


def _safe_redis_url(url: str) -> str:
    """Return Redis URL with credentials stripped (scheme://host:port only)."""
    p = urlparse(url)
    return f"{p.scheme}://{p.hostname}:{p.port}"


async def _connect_redis_with_retry(app: FastAPI) -> None:
    """Background task: connect to Redis with exponential backoff, no crash on failure."""
    delay = 5
    max_delay = 60
    while True:
        try:
            logger.info("[app] connecting to Redis at %s", _safe_redis_url(settings.redis_url))
            pool = await create_pool(build_redis_settings())
            app.state.redis = pool
            logger.info("[app] Redis pool ready")
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "[app] Redis connection failed (%s: %s), retrying in %ds",
                type(exc).__name__,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: launch Redis connection in background. Shutdown: clean up."""
    try:
        diag = get_playwright_runtime_diagnostics()
        logger.info(
            "[playwright] PLAYWRIGHT_BROWSERS_PATH=%s",
            diag.get("PLAYWRIGHT_BROWSERS_PATH", ""),
        )
        if diag.get("chromium_executable_path"):
            logger.info(
                "[playwright] chromium_executable_path=%s exists=%s",
                diag.get("chromium_executable_path", ""),
                diag.get("chromium_executable_exists", ""),
            )
        else:
            logger.info("[playwright] chromium_executable_path=<unknown>")
    except Exception as exc:
        logger.warning("[playwright] diagnostics failed: %s", exc)

    app.state.redis = None
    retry_task = asyncio.create_task(_connect_redis_with_retry(app))
    app.state.redis_retry_task = retry_task

    yield

    logger.info("[app] shutdown — cancelling Redis retry task")
    retry_task.cancel()
    try:
        await retry_task
    except asyncio.CancelledError:
        pass

    if app.state.redis is not None:
        logger.info("[app] shutdown — closing Redis pool")
        await app.state.redis.close()


app = FastAPI(
    title="Gradenza API",
    description="Backend API for OCR, grading, and material generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(analytics.router)
app.include_router(canvas_chat.router)
app.include_router(class_pdf_notes.router)
app.include_router(class_pdf_notes.legacy_router)
app.include_router(materials.router)
app.include_router(ocr.router)
app.include_router(online_lessons.router)
app.include_router(quiz.router)
app.include_router(submissions.router)


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
