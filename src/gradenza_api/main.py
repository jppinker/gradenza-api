"""
Gradenza API — FastAPI application entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gradenza_api.routes import analytics, canvas_chat, class_pdf_notes, materials, ocr, quiz, submissions
from gradenza_api.services.pdf_playwright import get_playwright_runtime_diagnostics
from gradenza_api.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: create Redis pool. Shutdown: close it."""
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

    logger.info("[app] startup — connecting to Redis at %s", settings.redis_url)
    app.state.redis = await create_pool(
        RedisSettings.from_dsn(settings.redis_url)
    )
    logger.info("[app] Redis pool ready")
    yield
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
app.include_router(quiz.router)
app.include_router(submissions.router)


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}
