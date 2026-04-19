"""
ARQ worker process.

Run with:
  arq gradenza_api.worker.WorkerSettings

Or via the Procfile:
  python -m gradenza_api.worker
"""

from __future__ import annotations

import asyncio
import logging

from arq.connections import RedisSettings

from gradenza_api.jobs.process_submission import process_submission
from gradenza_api.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


class WorkerSettings:
    """ARQ worker configuration."""

    functions = [process_submission]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    # How many jobs to run concurrently
    max_jobs = 5

    # Maximum seconds a job may run before being cancelled
    job_timeout = 600  # 10 minutes

    # Keep job results in Redis for 1 hour (useful for debugging)
    keep_result = 3600

    # Retry failed jobs (not including jobs that raised — those need explicit retry)
    max_tries = 3

    @staticmethod
    async def on_startup(ctx: dict) -> None:
        logger.info("[worker] startup")

    @staticmethod
    async def on_shutdown(ctx: dict) -> None:
        logger.info("[worker] shutdown")


if __name__ == "__main__":
    import arq

    asyncio.run(arq.run_worker(WorkerSettings))  # type: ignore[attr-defined]
