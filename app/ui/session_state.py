"""Derive Chainlit session flags from the graph's pending interrupt.

On resume, `cl.user_session` is empty but the LangGraph checkpoint knows what
the conversation is waiting for. This pure mapping turns the checkpoint-derived
`PendingInput` into the `awaiting` flag (plus narrowing context) so the next
user reply routes correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.graph.bridge import PendingInput


@dataclass(frozen=True)
class SessionState:
    awaiting: str | None
    narrowing_attribute: str | None = None
    narrowing_candidate_count: int = 0


def derive_session_state(pending: PendingInput | None) -> SessionState:
    """Map a pending interrupt to the Chainlit `awaiting` session state.

    `None` (no interrupt — e.g. a finished or fresh thread) falls back to
    `identity` so a resumed thread is always usable.
    """
    if pending is None:
        return SessionState(awaiting="identity")
    if pending.kind == "ask_narrowing":
        return SessionState(
            awaiting="narrowing",
            narrowing_attribute=pending.attribute,
            narrowing_candidate_count=pending.candidate_count,
        )
    if pending.kind == "confirm_profile":
        return SessionState(awaiting="confirm")
    return SessionState(awaiting="identity")
