"""Tests for the Chainlit chat-history data layer + schema bootstrap."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from app.db.chat_history import build_data_layer, init_chat_history_db

_EXPECTED_TABLES = {"users", "threads", "steps", "elements", "feedbacks"}


async def _table_names(db_path: Path) -> set[str]:
    async with aiosqlite.connect(db_path) as conn:
        rows = await (
            await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    return {r[0] for r in rows}


async def _column_names(db_path: Path, table: str) -> set[str]:
    async with aiosqlite.connect(db_path) as conn:
        rows = await (await conn.execute(f"PRAGMA table_info({table})")).fetchall()
    return {r[1] for r in rows}  # r[1] = column name


@pytest.mark.asyncio
async def test_init_creates_all_chainlit_tables(monkeypatch, tmp_path):
    db_path = tmp_path / "chat_history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db_path))
    from app.config import get_settings

    get_settings.cache_clear()

    await init_chat_history_db()

    assert _EXPECTED_TABLES.issubset(await _table_names(db_path))


@pytest.mark.asyncio
async def test_init_is_idempotent(monkeypatch, tmp_path):
    db_path = tmp_path / "chat_history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db_path))
    from app.config import get_settings

    get_settings.cache_clear()

    await init_chat_history_db()
    await init_chat_history_db()  # must not raise

    assert _EXPECTED_TABLES.issubset(await _table_names(db_path))


@pytest.mark.asyncio
async def test_init_enables_wal(monkeypatch, tmp_path):
    db_path = tmp_path / "chat_history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db_path))
    from app.config import get_settings

    get_settings.cache_clear()

    await init_chat_history_db()

    async with aiosqlite.connect(db_path) as conn:
        mode = await (await conn.execute("PRAGMA journal_mode")).fetchone()
    assert mode is not None and mode[0].lower() == "wal"


@pytest.mark.asyncio
async def test_schema_has_all_columns_chainlit_writes(monkeypatch, tmp_path):
    """The schema must declare every column SQLAlchemyDataLayer may emit.

    Chainlit builds INSERTs dynamically from the present step/element fields,
    and ``execute_sql`` swallows SQLAlchemy errors (logs a warning, returns
    None). So a column the data layer writes but the schema omits causes the
    step/element to be *silently* dropped — chat history persists incompletely
    with no visible error. These four columns are the ones Chainlit 2.11.x can
    write that are easy to miss.
    """
    db_path = tmp_path / "chat_history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db_path))
    from app.config import get_settings

    get_settings.cache_clear()

    await init_chat_history_db()

    assert {"command", "defaultOpen"} <= await _column_names(db_path, "steps")
    assert {"autoPlay", "playerConfig"} <= await _column_names(db_path, "elements")


def test_build_data_layer_points_at_configured_sqlite(monkeypatch, tmp_path):
    db_path = tmp_path / "chat_history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db_path))
    from app.config import get_settings

    get_settings.cache_clear()

    layer = build_data_layer()
    # async SQLite URL pointing at the configured file, no binary element storage
    assert layer._conninfo.startswith("sqlite+aiosqlite:///")
    assert "chat_history.db" in layer._conninfo
    assert layer.storage_provider is None
