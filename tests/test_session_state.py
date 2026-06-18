"""Restoring `awaiting` session state from a graph snapshot on resume."""

from __future__ import annotations

from app.graph.bridge import PendingInput
from app.ui.session_state import SessionState, derive_session_state


def test_none_falls_back_to_identity() -> None:
    # A resumed thread with no pending interrupt (e.g. done) → start fresh.
    assert derive_session_state(None) == SessionState(awaiting="identity")


def test_ask_identity() -> None:
    pending = PendingInput("ask_identity", "q", {}, 0, None)
    assert derive_session_state(pending) == SessionState(awaiting="identity")


def test_ask_narrowing_carries_attribute_and_count() -> None:
    pending = PendingInput("ask_narrowing", "q", {}, 3, "city")
    assert derive_session_state(pending) == SessionState(
        awaiting="narrowing", narrowing_attribute="city", narrowing_candidate_count=3
    )


def test_confirm_profile() -> None:
    pending = PendingInput("confirm_profile", "q", {}, 0, None)
    assert derive_session_state(pending) == SessionState(awaiting="confirm")
