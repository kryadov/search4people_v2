"""Tests for the A2A executor and its message helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from a2a.types import DataPart, Message, Part, Role, TextPart

from app.a2a import executor as ex
from app.graph.bridge import PendingInput


def _msg(parts: list[Part], metadata: dict | None = None) -> Message:
    return Message(role=Role.user, parts=parts, message_id="m1", metadata=metadata)


def test_message_to_identity_from_text():
    msg = _msg([Part(root=TextPart(text="Jane Doe"))])
    assert ex.message_to_identity(msg) == {"first_name": "Jane", "last_name": "Doe"}


def test_message_to_identity_from_data():
    msg = _msg([Part(root=DataPart(data={"first_name": "Ann", "last_name": "Lee", "city": "Berlin"}))])
    ident = ex.message_to_identity(msg)
    assert ident["first_name"] == "Ann"
    assert ident["city"] == "Berlin"


def test_message_to_answer_pick_index():
    pending = PendingInput(kind="ask_narrowing", question="", data={}, candidate_count=3, attribute="city")
    msg = _msg([Part(root=DataPart(data={"pick_index": 2}))])
    ans = ex.message_to_answer(msg, pending)
    assert ans.pick_index == 2


def test_message_to_answer_attribute_value():
    pending = PendingInput(kind="ask_narrowing", question="", data={}, candidate_count=3, attribute="city")
    msg = _msg([Part(root=DataPart(data={"attribute": "city", "value": "Berlin"}))])
    ans = ex.message_to_answer(msg, pending)
    assert ans.attribute_value == ("city", "Berlin")


def test_message_to_answer_narrowing_freetext():
    pending = PendingInput(kind="ask_narrowing", question="", data={}, candidate_count=3, attribute=None)
    msg = _msg([Part(root=TextPart(text="works at ACME"))])
    ans = ex.message_to_answer(msg, pending)
    assert ans.extra == "works at ACME"


def test_message_to_answer_confirm_decision():
    pending = PendingInput(kind="confirm_profile", question="", data={}, candidate_count=0, attribute=None)
    msg = _msg([Part(root=DataPart(data={"decision": "approve"}))])
    assert ex.message_to_answer(msg, pending).decision == "approve"


def test_message_to_answer_confirm_text_yes():
    pending = PendingInput(kind="confirm_profile", question="", data={}, candidate_count=0, attribute=None)
    msg = _msg([Part(root=TextPart(text="yes"))])
    assert ex.message_to_answer(msg, pending).decision == "approve"


def test_pending_to_parts_includes_data():
    pending = PendingInput(
        kind="ask_narrowing",
        question="Which city?",
        data={"candidates": [{"url": "u1"}], "options": ["Berlin"], "attribute": "city"},
        candidate_count=1,
        attribute="city",
    )
    parts = ex.pending_to_parts(pending)
    kinds = [p.root.kind for p in parts]
    assert "text" in kinds and "data" in kinds


def test_locale_from_message_defaults_en():
    assert ex.locale_from_message(_msg([Part(root=TextPart(text="x"))])) == "en"
    assert ex.locale_from_message(_msg([Part(root=TextPart(text="x"))], {"locale": "ru"})) == "ru"


class _RecordingQueue:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


def _ctx(message, current_task=None):
    return SimpleNamespace(
        message=message,
        current_task=current_task,
        task_id=None,
        context_id=None,
    )


@pytest.mark.asyncio
async def test_execute_completes_with_profile(monkeypatch):
    # Stub the graph: no pending interrupt, phase done, profile present.
    done_state = {
        "phase": "done",
        "user_decision": "approve",
        "profile": {"full_name": "Jane Doe", "confidence": "medium", "evidence": []},
    }

    class FakeGraph:
        async def aget_state(self, config):
            return SimpleNamespace(tasks=[], values=done_state)

        async def astream(self, inp, config, stream_mode):
            if False:
                yield {}
            return

    saved = {}

    async def fake_save_profile(**kwargs):
        saved.update(kwargs)
        return 1

    monkeypatch.setattr(ex, "save_profile", fake_save_profile)

    executor = ex.PeopleSearchExecutor(FakeGraph(), current_user_id=lambda: 42)
    queue = _RecordingQueue()
    msg = _msg([Part(root=TextPart(text="Jane Doe"))])
    await executor.execute(_ctx(msg), queue)

    # The profile was persisted under the authenticated user.
    assert saved["user_id"] == 42
    assert saved["full_name"] == "Jane Doe"
    # An artifact and a terminal status were enqueued.
    assert queue.events  # at least task + status/artifact events recorded


@pytest.mark.asyncio
async def test_cancel_aborts_graph_when_on_confirm(monkeypatch):
    resumed_with = {}

    class FakeGraph:
        async def aget_state(self, config):
            interrupt = SimpleNamespace(value={"kind": "confirm_profile", "profile": {}, "candidates": [], "locale": "en"})
            task = SimpleNamespace(interrupts=[interrupt])
            return SimpleNamespace(tasks=[task], values={})

        async def astream(self, inp, config, stream_mode):
            resumed_with["resume"] = inp.resume  # Command.resume
            if False:
                yield {}
            return

    executor = ex.PeopleSearchExecutor(FakeGraph(), current_user_id=lambda: 1)
    queue = _RecordingQueue()
    pre_task = SimpleNamespace(id="task-x", context_id="ctx-x")
    ctx = SimpleNamespace(message=_msg([Part(root=TextPart(text="stop"))]), current_task=pre_task)
    await executor.cancel(ctx, queue)
    assert resumed_with.get("resume") == {"decision": "abort"}
