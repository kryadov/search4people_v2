"""Shared pytest fixtures.

We pin a temporary SQLite path and provide minimal env so app.config doesn't
choke when imported during collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "test-secret-must-be-long-enough")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "")  # force DDG-only path in tests
    monkeypatch.setenv("SEARCH_PROVIDERS", "ddg")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    # Drop the lru_cached settings singleton between tests.
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
