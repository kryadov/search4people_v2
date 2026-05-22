"""SQLite connection helpers + idempotent migration runner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from app.config import get_settings

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


async def init_db() -> None:
    """Create the database file (if missing) and apply the schema."""
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.executescript(sql)
        await conn.commit()


@asynccontextmanager
async def connect() -> AsyncIterator[aiosqlite.Connection]:
    """Open an async SQLite connection with foreign keys enabled."""
    settings = get_settings()
    async with aiosqlite.connect(settings.db_path) as conn:
        await conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = aiosqlite.Row
        yield conn
