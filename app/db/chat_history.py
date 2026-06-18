"""Persistent chat-history storage: Chainlit data-layer DB + factory.

Lives in its own SQLite file (`settings.chat_history_db_path`) because
Chainlit's data-layer schema defines a `users` table that collides with the
auth `users` table in `app.db`.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from app.config import get_settings

_SCHEMA_PATH = Path(__file__).with_name("chat_history_schema.sql")


async def init_chat_history_db() -> None:
    """Create the chat-history DB (if missing), apply the schema, enable WAL."""
    settings = get_settings()
    settings.chat_history_db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    async with aiosqlite.connect(settings.chat_history_db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.executescript(sql)
        await conn.commit()
