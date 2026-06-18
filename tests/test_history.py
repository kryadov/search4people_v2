"""Tests for the pure chat-history session helpers (no Chainlit/DB)."""

from __future__ import annotations

from types import SimpleNamespace

from app.history import (
    build_fresh_input,
    compute_thread_tags,
    derive_awaiting,
    render_tag_line,
    resume_awaiting,
)
from app.models.profile import Evidence, PersonProfile


def _snapshot_with_interrupt(payload: dict) -> SimpleNamespace:
    interrupt = SimpleNamespace(value=payload)
    task = SimpleNamespace(interrupts=[interrupt])
    return SimpleNamespace(tasks=[task], values={})


# ----- build_fresh_input -----


def test_build_fresh_input_resets_transient_keys():
    state = build_fresh_input({"first_name": "Jane", "last_name": "Doe"}, "ru")
    assert state["query"] == {"first_name": "Jane", "last_name": "Doe"}
    assert state["locale"] == "ru"
    # Transient search state must be cleared so a follow-up on a done thread
    # does not inherit a stale iteration count / profile / candidates.
    assert state["iteration"] == 0
    assert state["candidates"] == []
    assert state["visited_urls"] == []
    assert state["fetched_pages"] == []
    assert state["profile"] is None
    assert state["user_decision"] is None
    assert state["selected_candidate_index"] is None
    assert state["guard_block"] is None
    # We must NOT pin max_iterations (it is config, defaulted by the node).
    assert "max_iterations" not in state


# ----- derive_awaiting -----


def test_derive_awaiting_identity():
    snap = _snapshot_with_interrupt({"kind": "ask_identity", "locale": "en"})
    result = derive_awaiting(snap, "en")
    assert result.awaiting == "identity"


def test_derive_awaiting_narrowing_carries_attribute_and_count():
    snap = _snapshot_with_interrupt(
        {
            "kind": "ask_narrowing",
            "attribute": "city",
            "question": "Which city?",
            "candidates": [{"url": "u1"}, {"url": "u2"}, {"url": "u3"}],
            "locale": "en",
        }
    )
    result = derive_awaiting(snap, "en")
    assert result.awaiting == "narrowing"
    assert result.narrowing_attribute == "city"
    assert result.narrowing_candidate_count == 3


def test_derive_awaiting_confirm():
    snap = _snapshot_with_interrupt(
        {"kind": "confirm_profile", "profile": {"full_name": "X"}, "candidates": [], "locale": "en"}
    )
    assert derive_awaiting(snap, "en").awaiting == "confirm"


def test_derive_awaiting_none_when_no_interrupt():
    snap = SimpleNamespace(tasks=[], values={})
    assert derive_awaiting(snap, "en").awaiting is None


# ----- resume_awaiting (fallback to identity for done/empty threads) -----


def test_resume_awaiting_falls_back_to_identity():
    snap = SimpleNamespace(tasks=[], values={})
    assert resume_awaiting(snap, "en").awaiting == "identity"


def test_resume_awaiting_preserves_active_interrupt():
    snap = _snapshot_with_interrupt(
        {"kind": "ask_narrowing", "attribute": "city", "candidates": [{"url": "u1"}], "locale": "en"}
    )
    result = resume_awaiting(snap, "en")
    assert result.awaiting == "narrowing"
    assert result.narrowing_attribute == "city"
    assert result.narrowing_candidate_count == 1


# ----- compute_thread_tags -----


def _profile_with(platforms: list[str | None], confidence: str = "medium") -> PersonProfile:
    evidence = [
        Evidence(url="https://example.com/x", platform=p, snippet=None) for p in platforms
    ]
    return PersonProfile(full_name="Jane Doe", confidence=confidence, evidence=evidence)


def test_compute_thread_tags_namespaced_sorted_deduped():
    profile = _profile_with(["LinkedIn", "github", "github"], confidence="high")
    tags = compute_thread_tags(profile, "ru")
    assert tags == ["platform:github", "platform:linkedin", "confidence:high", "locale:ru"]


def test_compute_thread_tags_skips_missing_platforms():
    profile = _profile_with([None, ""], confidence="low")
    assert compute_thread_tags(profile, "en") == ["confidence:low", "locale:en"]


# ----- render_tag_line -----


def test_render_tag_line_contains_all_tokens():
    line = render_tag_line(["platform:github", "confidence:high", "locale:ru"], "ru")
    assert "platform:github" in line
    assert "confidence:high" in line
    assert "locale:ru" in line
    # tokens are joined with a separator, not collapsed
    assert "·" in line
