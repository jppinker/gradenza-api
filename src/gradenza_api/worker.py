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
import time

from arq.connections import RedisSettings

from gradenza_api.jobs.generate_quiz import generate_quiz
from gradenza_api.jobs.process_submission import process_submission
from gradenza_api.settings import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

logger = logging.getLogger(__name__)


class WorkerSettings:
    """ARQ worker configuration."""

    functions = [generate_quiz, process_submission]
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


def _run_worker_once() -> None:
    """Run ARQ worker once, handling both sync and async run_worker variants."""
    import arq
    import inspect

    result = arq.run_worker(WorkerSettings)  # type: ignore[attr-defined]
    if inspect.isawaitable(result):
        asyncio.run(result)


def main() -> None:
    # Retry loop to protect against temporary Redis / Railway private-network
    # unavailability during container startup.  Railway sometimes takes a few
    # seconds to make the private network (redis.railway.internal) reachable
    # after a worker container starts, so we retry with exponential backoff
    # instead of letting the process die immediately.
    import redis.exceptions

    delay = 5
    max_delay = 60
    attempt = 0

    while True:
        attempt += 1
        try:
            logger.info("[worker] starting ARQ worker (attempt %d)", attempt)
            _run_worker_once()
            # run_worker returned cleanly — ARQ decided to stop, so we stop too.
            logger.info("[worker] ARQ worker exited cleanly")
            break
        except (
            redis.exceptions.ConnectionError,
            redis.exceptions.TimeoutError,
        ) as exc:
            logger.warning(
                "[worker] Redis unavailable on attempt %d (%s: %s); "
                "retrying in %ds",
                attempt, type(exc).__name__, exc, delay,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[worker] unexpected error on attempt %d (%s: %s); "
                "retrying in %ds",
                attempt, type(exc).__name__, exc, delay,
            )

        time.sleep(delay)
        delay = min(delay * 2, max_delay)


if __name__ == "__main__":
    main()
