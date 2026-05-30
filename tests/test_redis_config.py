"""Unit tests for build_redis_settings().

No network or live Redis required — only DSN parsing is exercised.
"""

from __future__ import annotations

import pytest

from gradenza_api.redis_config import build_redis_settings


def test_plain_redis_with_credentials_and_db():
    s = build_redis_settings("redis://default:pass@example.com:6379/0")
    assert s.host == "example.com"
    assert s.port == 6379
    assert s.username == "default"
    assert s.password == "pass"
    assert s.database == 0
    assert s.ssl is False


def test_rediss_enables_ssl_and_parses_db():
    s = build_redis_settings("rediss://default:pass@example.com:6380/1")
    assert s.host == "example.com"
    assert s.port == 6380
    assert s.username == "default"
    assert s.password == "pass"
    assert s.database == 1
    assert s.ssl is True


def test_no_credentials_no_db_defaults():
    s = build_redis_settings("redis://example.com:6379")
    assert s.host == "example.com"
    assert s.port == 6379
    assert s.username is None
    assert s.password is None
    assert s.database == 0
    assert s.ssl is False


def test_url_encoded_password_is_unquoted():
    s = build_redis_settings("redis://default:p%40ss%2Fword@example.com:6379/2")
    assert s.password == "p@ss/word"
    assert s.database == 2


def test_missing_url_raises():
    with pytest.raises(RuntimeError, match="REDIS_URL is not set"):
        build_redis_settings("")


def test_unsupported_scheme_raises():
    with pytest.raises(RuntimeError, match="Unsupported REDIS_URL scheme"):
        build_redis_settings("http://example.com:6379")
