"""Tests for the SQLite-backed A2A TaskStore."""

from __future__ import annotations

import pytest
from a2a.types import Task, TaskState, TaskStatus

from app.a2a.task_store import SqliteTaskStore
from app.db.connection import init_db


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        context_id="ctx-1",
        status=TaskStatus(state=TaskState.submitted),
    )


@pytest.mark.asyncio
async def test_save_and_get_roundtrip():
    await init_db()
    store = SqliteTaskStore()
    task = _task("task-1")
    await store.save(task)
    loaded = await store.get("task-1")
    assert loaded is not None
    assert loaded.id == "task-1"
    assert loaded.status.state == TaskState.submitted


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    await init_db()
    store = SqliteTaskStore()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_save_is_upsert():
    await init_db()
    store = SqliteTaskStore()
    await store.save(_task("task-2"))
    updated = _task("task-2")
    updated.status = TaskStatus(state=TaskState.completed)
    await store.save(updated)
    loaded = await store.get("task-2")
    assert loaded.status.state == TaskState.completed


@pytest.mark.asyncio
async def test_delete():
    await init_db()
    store = SqliteTaskStore()
    await store.save(_task("task-3"))
    await store.delete("task-3")
    assert await store.get("task-3") is None
