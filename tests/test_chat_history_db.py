"""Chat-history DB init + data-layer factory."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest


@pytest.mark.anyio
async def test_init_chat_history_db_creates_tables(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db))
    from app.config import get_settings

    get_settings.cache_clear()

    from app.db.chat_history import init_chat_history_db

    await init_chat_history_db()
    await init_chat_history_db()  # idempotent: second call must not raise

    async with aiosqlite.connect(db) as conn:
        rows = await (
            await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        mode = await (await conn.execute("PRAGMA journal_mode")).fetchone()

    names = {r[0] for r in rows}
    assert {"users", "threads", "steps", "elements", "feedbacks"} <= names
    assert mode is not None and mode[0].lower() == "wal"
