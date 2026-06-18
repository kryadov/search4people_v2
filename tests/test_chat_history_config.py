"""Config + dependency wiring for persistent chat history."""

from __future__ import annotations

from pathlib import Path


def test_chat_history_db_path_default(monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.delenv("CHAT_HISTORY_DB_PATH", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.chat_history_db_path == Path("data/chat_history.db")


def test_chat_history_db_path_env_override(monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", "/tmp/custom_history.db")
    get_settings.cache_clear()
    assert get_settings().chat_history_db_path == Path("/tmp/custom_history.db")


def test_sqlalchemy_importable() -> None:
    import sqlalchemy  # noqa: F401
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: F401
