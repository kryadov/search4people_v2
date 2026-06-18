"""Chainlit chat-history persistence: schema bootstrap + data-layer factory.

The Chainlit `SQLAlchemyDataLayer` does not create its own tables, so we apply
the schema (`chat_history_schema.sql`) idempotently at startup, mirroring
`app/db/connection.py::init_db`. It lives in its own SQLite file
(`settings.chat_history_db_path`) to avoid the `users` table colliding with the
auth `users` table in `db_path`.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

from app.config import get_settings

_SCHEMA_PATH = Path(__file__).with_name("chat_history_schema.sql")


async def init_chat_history_db() -> None:
    """Create the chat-history database (if missing), apply the schema, set WAL."""
    settings = get_settings()
    settings.chat_history_db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    async with aiosqlite.connect(settings.chat_history_db_path) as conn:
        # WAL lets the data-layer engine and the app's other connections write
        # concurrently without "database is locked" errors.
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.executescript(sql)
        await conn.commit()


def build_data_layer() -> SQLAlchemyDataLayer:
    """A `SQLAlchemyDataLayer` bound to the chat-history SQLite file.

    `storage_provider=None`: profile cards are plain markdown messages, so there
    are no binary elements to persist to object storage.
    """
    db_path = get_settings().chat_history_db_path.resolve()
    conninfo = f"sqlite+aiosqlite:///{db_path.as_posix()}"
    return SQLAlchemyDataLayer(conninfo=conninfo, storage_provider=None)
