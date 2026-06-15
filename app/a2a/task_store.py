"""A SQLite-backed A2A TaskStore.

Persists A2A Task objects in `data/app.db` so task metadata survives a server
restart. The resumable conversation state itself lives in the LangGraph
checkpointer (keyed by thread_id == task_id); this store only holds the A2A
protocol view of each task for `tasks/get` and resubscription.
"""

from __future__ import annotations

from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import Task

from app.db.connection import connect


class SqliteTaskStore(TaskStore):
    async def save(
        self, task: Task, context: ServerCallContext | None = None
    ) -> None:
        payload = task.model_dump_json()
        async with connect() as conn:
            await conn.execute(
                "INSERT INTO a2a_tasks (task_id, task_json, updated_at) "
                "VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(task_id) DO UPDATE SET "
                "task_json = excluded.task_json, updated_at = datetime('now')",
                (task.id, payload),
            )
            await conn.commit()

    async def get(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> Task | None:
        async with connect() as conn:
            row = await (
                await conn.execute(
                    "SELECT task_json FROM a2a_tasks WHERE task_id = ?", (task_id,)
                )
            ).fetchone()
        if row is None:
            return None
        return Task.model_validate_json(row["task_json"])

    async def delete(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> None:
        async with connect() as conn:
            await conn.execute(
                "DELETE FROM a2a_tasks WHERE task_id = ?", (task_id,)
            )
            await conn.commit()
