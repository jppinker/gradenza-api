"""
Shared Redis / ARQ connection configuration.

A single source of truth for building :class:`arq.connections.RedisSettings`
from the ``REDIS_URL`` environment variable.  This is used by every place
that connects to Redis (the API enqueue pool, the ARQ worker, and any health
check) so that switching between Railway's public and private Redis URLs is a
pure environment-variable change — no code change required.

Railway exposes two URLs for a Redis service:

  * Private:  ``REDIS_URL=${{Redis.REDIS_URL}}``         (redis.railway.internal)
  * Public:   ``REDIS_URL=${{Redis.REDIS_PUBLIC_URL}}``  (xxx.proxy.rlwy.net)

Both are standard ``redis://`` (or ``rediss://``) DSNs that include username,
password, host, port and optionally a database number.  Because we parse and
preserve every component, either value can be dropped into ``REDIS_URL``
without touching the code.
"""

from __future__ import annotations

import logging
from urllib.parse import unquote, urlparse

from arq.connections import RedisSettings

from gradenza_api.settings import settings

logger = logging.getLogger(__name__)


def build_redis_settings(redis_url: str | None = None) -> RedisSettings:
    """Build a fully-populated :class:`RedisSettings` from ``REDIS_URL``.

    Parses and preserves the scheme, hostname, port, username, password,
    database number and SSL/TLS flag.  We deliberately do **not** construct
    ``RedisSettings`` from only host/port, as that would silently drop the
    credentials and break authenticated Railway Redis.

    :param redis_url: optional explicit DSN; defaults to ``settings.redis_url``
        (which is populated from the ``REDIS_URL`` environment variable / .env).
    :raises RuntimeError: if no URL is configured or the scheme is unsupported.
    """
    redis_url = redis_url if redis_url is not None else settings.redis_url
    if not redis_url:
        raise RuntimeError("REDIS_URL is not set")

    parsed = urlparse(redis_url)

    if parsed.scheme not in {"redis", "rediss"}:
        raise RuntimeError(f"Unsupported REDIS_URL scheme: {parsed.scheme!r}")

    db_raw = (parsed.path or "/0").lstrip("/") or "0"
    try:
        database = int(db_raw)
    except ValueError:
        database = 0

    use_ssl = parsed.scheme == "rediss"

    # Safe diagnostics: never log the full URL or the password.
    logger.info(
        "[redis] config scheme=%s host=%s port=%s username=%s has_password=%s db=%s ssl=%s",
        parsed.scheme,
        parsed.hostname,
        parsed.port,
        parsed.username,
        bool(parsed.password),
        database,
        use_ssl,
    )

    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        username=unquote(parsed.username) if parsed.username else None,
        password=unquote(parsed.password) if parsed.password else None,
        database=database,
        ssl=use_ssl,
    )
