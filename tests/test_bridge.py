"""Tests for the headless interrupt/resume bridge."""

from __future__ import annotations

from types import SimpleNamespace

from langgraph.types import Command

from app.graph.bridge import (
    PendingInput,
    ResumeAnswer,
    build_resume_command,
    parse_identity_text,
    read_pending_input,
)


def _snapshot_with_interrupt(payload: dict) -> SimpleNamespace:
    interrupt = SimpleNamespace(value=payload)
    task = SimpleNamespace(interrupts=[interrupt])
    return SimpleNamespace(tasks=[task], values={})


def test_read_pending_identity():
    snap = _snapshot_with_interrupt({"kind": "ask_identity", "locale": "en"})
    pending = read_pending_input(snap, "en")
    assert pending is not None
    assert pending.kind == "ask_identity"
    assert pending.candidate_count == 0


def test_read_pending_narrowing():
    snap = _snapshot_with_interrupt(
        {
            "kind": "ask_narrowing",
            "attribute": "city",
            "question": "Which city?",
            "options": ["Berlin", "Munich"],
            "candidates": [{"url": "u1"}, {"url": "u2"}],
            "locale": "en",
        }
    )
    pending = read_pending_input(snap, "en")
    assert pending.kind == "ask_narrowing"
    assert pending.attribute == "city"
    assert pending.question == "Which city?"
    assert pending.candidate_count == 2
    assert pending.data["options"] == ["Berlin", "Munich"]


def test_read_pending_confirm():
    snap = _snapshot_with_interrupt(
        {"kind": "confirm_profile", "profile": {"full_name": "X"}, "candidates": [], "locale": "en"}
    )
    pending = read_pending_input(snap, "en")
    assert pending.kind == "confirm_profile"
    assert pending.data["profile"] == {"full_name": "X"}


def test_read_pending_none_when_no_interrupt():
    snap = SimpleNamespace(tasks=[], values={})
    assert read_pending_input(snap, "en") is None


def test_build_resume_identity():
    pending = PendingInput(kind="ask_identity", question="", data={}, candidate_count=0, attribute=None)
    cmd = build_resume_command(pending, ResumeAnswer(identity={"first_name": "Jane", "last_name": "Doe"}))
    assert isinstance(cmd, Command)
    assert cmd.resume == {"first_name": "Jane", "last_name": "Doe"}


def test_build_resume_pick_index():
    pending = PendingInput(kind="ask_narrowing", question="", data={}, candidate_count=3, attribute="city")
    cmd = build_resume_command(pending, ResumeAnswer(pick_index=1))
    assert cmd.resume == {"pick_index": 1}


def test_build_resume_attribute_value():
    pending = PendingInput(kind="ask_narrowing", question="", data={}, candidate_count=3, attribute="city")
    cmd = build_resume_command(pending, ResumeAnswer(attribute_value=("city", "Berlin")))
    assert cmd.resume == {"attribute": "city", "value": "Berlin"}


def test_build_resume_extra():
    pending = PendingInput(kind="ask_narrowing", question="", data={}, candidate_count=3, attribute=None)
    cmd = build_resume_command(pending, ResumeAnswer(extra="works at ACME"))
    assert cmd.resume == {"extra": "works at ACME"}


def test_build_resume_confirm_decision():
    pending = PendingInput(kind="confirm_profile", question="", data={}, candidate_count=0, attribute=None)
    assert build_resume_command(pending, ResumeAnswer(decision="approve")).resume == {"decision": "approve"}
    assert build_resume_command(pending, ResumeAnswer(decision="abort")).resume == {"decision": "abort"}


def test_build_resume_confirm_decision_with_extra():
    pending = PendingInput(kind="confirm_profile", question="", data={}, candidate_count=0, attribute=None)
    cmd = build_resume_command(pending, ResumeAnswer(decision="more", extra="add education"))
    assert cmd.resume == {"decision": "more", "extra": "add education"}


def test_parse_identity_text():
    assert parse_identity_text("Jane") == {"first_name": "Jane"}
    assert parse_identity_text("Jane Doe Smith") == {"first_name": "Jane", "last_name": "Doe Smith"}
    assert parse_identity_text("   ") == {}


def test_fresh_search_input_resets_transient_state() -> None:
    from app.graph.bridge import fresh_search_input

    out = fresh_search_input({"first_name": "Jane", "last_name": "Doe"}, "ru")
    assert out["query"] == {"first_name": "Jane", "last_name": "Doe"}
    assert out["locale"] == "ru"
    assert out["phase"] == "collect"
    assert out["candidates"] == []
    assert out["fetched_pages"] == []
    assert out["visited_urls"] == []
    assert out["iteration"] == 0
    assert out["profile"] is None
    assert out["user_decision"] is None
    assert out["selected_candidate_index"] is None
    assert out["guard_block"] is None
    # messages must NOT be reset (add_messages reducer preserves history).
    assert "messages" not in out
