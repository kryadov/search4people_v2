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

    assert {"defaultOpen", "autoCollapse"} <= await _column_names(db_path, "steps")
    assert {"autoPlay", "playerConfig"} <= await _column_names(db_path, "elements")


# Every key Chainlit's Step.to_dict() emits on each create_step. A non-None
# value that has no column makes the INSERT raise (and execute_sql swallows it,
# silently dropping the step). Kept in sync with chainlit/step.py::to_dict.
_STEP_TO_DICT_COLUMNS = [
    "id", "name", "type", "threadId", "parentId", "streaming", "metadata",
    "tags", "input", "isError", "output", "createdAt", "start", "end",
    "language", "defaultOpen", "autoCollapse", "showInput", "generation",
]


@pytest.mark.asyncio
async def test_steps_table_accepts_every_step_to_dict_column(monkeypatch, tmp_path):
    """A real INSERT of all Step.to_dict() columns must not raise.

    This is the robust guard: it fails for ANY column Chainlit writes but the
    schema omits, without us having to enumerate them by hand.
    """
    db_path = tmp_path / "chat_history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db_path))
    from app.config import get_settings

    get_settings.cache_clear()
    await init_chat_history_db()

    columns = ", ".join(f'"{c}"' for c in _STEP_TO_DICT_COLUMNS)
    placeholders = ", ".join("?" for _ in _STEP_TO_DICT_COLUMNS)
    # streaming is NOT NULL; the rest can be dummy/empty values.
    values = [
        "step-1", "msg", "user_message", "thread-1", None, 0, "{}",
        "[]", "", 0, "", "2026-01-01", "", "", "en", 0, 0, "false", None,
    ]
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            f"INSERT INTO steps ({columns}) VALUES ({placeholders})", values
        )
        await conn.commit()
        row = await (await conn.execute('SELECT "id" FROM steps')).fetchone()
    assert row is not None and row[0] == "step-1"


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
