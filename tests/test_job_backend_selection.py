"""How the job backend is chosen.

Getting this wrong is expensive in a quiet way: a production deployment that
silently falls back to a single-process thread pool looks healthy while losing
every queued job on restart. Hence three explicit modes and a backend name in
``/health``.
"""

from __future__ import annotations

import pytest

from frontier.jobs import create_job_store
from frontier.settings import get_settings


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch):
    """Set SO_* environment variables and rebuild the cached Settings."""

    def apply(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(f"SO_{key.upper()}", value)
        get_settings.cache_clear()
        return get_settings()

    yield apply
    get_settings.cache_clear()


def test_memory_mode_never_touches_redis(configure):
    configure(job_backend="memory", redis_url="redis://127.0.0.1:1/0")
    store = create_job_store()
    try:
        assert store.backend == "memory"
    finally:
        store.shutdown()


def test_auto_mode_falls_back_when_redis_is_unreachable(configure, caplog):
    # Port 1 is reserved and will refuse immediately.
    configure(job_backend="auto", redis_url="redis://127.0.0.1:1/0")

    with caplog.at_level("WARNING"):
        store = create_job_store()
    try:
        assert store.backend == "memory"
        # The fallback must be loud in the log even though it is not fatal.
        assert any("falling back" in record.message for record in caplog.records)
    finally:
        store.shutdown()


def test_redis_mode_refuses_to_start_without_redis(configure):
    """Production should fail fast rather than degrade quietly."""
    configure(job_backend="redis", redis_url="redis://127.0.0.1:1/0")

    with pytest.raises(RuntimeError, match="unreachable"):
        create_job_store()


def test_settings_are_overridable_from_the_environment(configure):
    settings = configure(
        job_backend="memory",
        redis_url="redis://example:6379/3",
        queue_name="custom",
        job_timeout_seconds="600",
    )
    assert settings.redis_url == "redis://example:6379/3"
    assert settings.queue_name == "custom"
    assert settings.job_timeout_seconds == 600
