"""Headless interrupt/resume bridge.

Protocol-neutral translation between LangGraph `interrupt()` payloads and the
`Command(resume=...)` values the graph nodes expect. Used by BOTH the Chainlit
UI (`app/main.py`) and the A2A server (`app/a2a/executor.py`). Knows nothing
about Chainlit or A2A — each frontend parses its own raw input into a
`ResumeAnswer` and renders `PendingInput` in its own format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langgraph.types import Command

from app.i18n import Locale, t

PendingKind = Literal["ask_identity", "ask_narrowing", "confirm_profile"]


@dataclass
class PendingInput:
    """A normalized view of the interrupt the graph is currently paused on."""

    kind: PendingKind
    question: str
    data: dict[str, Any]
    candidate_count: int
    attribute: str | None


@dataclass
class ResumeAnswer:
    """A frontend-parsed answer. Exactly one intent is set, matching the kind."""

    identity: dict[str, Any] | None = None          # ask_identity
    pick_index: int | None = None                   # ask_narrowing
    attribute_value: tuple[str, str] | None = None  # ask_narrowing
    extra: str | None = None                        # ask_narrowing
    decision: str | None = None                     # confirm_profile


def parse_identity_text(text: str) -> dict[str, Any]:
    """First token → first_name, remainder → last_name (heuristic)."""
    tokens = [tok for tok in (text or "").strip().split() if tok]
    if not tokens:
        return {}
    if len(tokens) == 1:
        return {"first_name": tokens[0]}
    return {"first_name": tokens[0], "last_name": " ".join(tokens[1:])}


def fresh_search_input(identity: dict[str, Any], locale: Locale) -> dict[str, Any]:
    """A full state reset for a brand-new search on an existing thread.

    Used when a finished (`phase == "done"`) thread receives a new identity:
    the conversation continues in the same thread, but every plain (non-reduced)
    state channel is overwritten so the new search does not inherit stale
    candidates/profile. `messages` is omitted on purpose — its `add_messages`
    reducer must keep the prior history visible.
    """
    return {
        "query": identity,
        "locale": locale,
        "candidates": [],
        "visited_urls": [],
        "fetched_pages": [],
        "iteration": 0,
        "phase": "collect",
        "profile": None,
        "user_decision": None,
        "selected_candidate_index": None,
        "guard_block": None,
    }


def read_pending_input(snapshot: Any, locale: Locale) -> PendingInput | None:
    """Pull the active interrupt out of a LangGraph state snapshot."""
    tasks = getattr(snapshot, "tasks", None) or []
    for task in tasks:
        for itr in getattr(task, "interrupts", None) or []:
            payload = getattr(itr, "value", itr)
            if not isinstance(payload, dict):
                continue
            kind = payload.get("kind")
            if kind == "ask_identity":
                return PendingInput(
                    kind="ask_identity",
                    question=t("ask_who", locale),
                    data={},
                    candidate_count=0,
                    attribute=None,
                )
            if kind == "ask_narrowing":
                candidates = list(payload.get("candidates") or [])
                return PendingInput(
                    kind="ask_narrowing",
                    question=payload.get("question") or "",
                    data={
                        "candidates": candidates,
                        "options": list(payload.get("options") or []),
                        "attribute": payload.get("attribute"),
                    },
                    candidate_count=len(candidates),
                    attribute=payload.get("attribute"),
                )
            if kind == "confirm_profile":
                candidates = list(payload.get("candidates") or [])
                return PendingInput(
                    kind="confirm_profile",
                    question=t("confirm_profile", locale),
                    data={"profile": payload.get("profile"), "candidates": candidates},
                    candidate_count=len(candidates),
                    attribute=None,
                )
    return None


def build_resume_command(pending: PendingInput, answer: ResumeAnswer) -> Command:
    """Map a parsed answer onto the exact resume dict each node expects."""
    if pending.kind == "ask_identity":
        return Command(resume=dict(answer.identity or {}))
    if pending.kind == "ask_narrowing":
        if answer.pick_index is not None:
            return Command(resume={"pick_index": answer.pick_index})
        if answer.attribute_value is not None:
            attr, value = answer.attribute_value
            return Command(resume={"attribute": attr, "value": value})
        return Command(resume={"extra": answer.extra or ""})
    if pending.kind == "confirm_profile":
        resume: dict[str, Any] = {"decision": answer.decision or "approve"}
        if answer.extra:
            resume["extra"] = answer.extra
        return Command(resume=resume)
    raise ValueError(f"Unknown pending kind: {pending.kind!r}")
