"""Append-only audit log of guardrail decisions (one row per fired finding)."""

from __future__ import annotations

import structlog

from app.db.connection import connect
from app.guardrails.types import GuardVerdict

log = structlog.get_logger()

_SNIPPET_MAX = 200


def _safe_snippet(text: str) -> str:
    """Truncate; the caller passes already-redacted text so logs cannot leak PII."""
    s = text.strip().replace("\n", " ")
    return s[:_SNIPPET_MAX]


async def record_events(
    verdict: GuardVerdict,
    *,
    point: str,
    snippet_source: str,
    thread_id: str | None = None,
    user_id: int | None = None,
) -> None:
    """Best-effort: write one guard_events row per finding. Never raises."""
    if not verdict.findings:
        return
    snippet = _safe_snippet(verdict.transformed_text or snippet_source)
    rows = [
        (
            user_id,
            thread_id,
            point,
            f.category,
            verdict.action,
            f.score,
            f.label,
            snippet,
            verdict.action,
        )
        for f in verdict.findings
    ]
    try:
        async with connect() as conn:
            await conn.executemany(
                "INSERT INTO guard_events "
                "(user_id, thread_id, point, category, action, score, label, snippet, decision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            await conn.commit()
    except Exception as exc:  # audit must never break the request
        log.warning("guard_audit_failed", error=str(exc))
