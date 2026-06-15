"""A2A executor: drives the people-search graph for an A2A task.

Maps the graph's `interrupt()` pauses onto the A2A `input-required` state and
the final PersonProfile onto a completed-task artifact. Pure helpers
(`message_to_identity`, `message_to_answer`, `pending_to_parts`,
`locale_from_message`) are unit-tested directly.
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import new_task

from app.db.profiles import save_profile
from app.graph.bridge import (
    PendingInput,
    ResumeAnswer,
    build_resume_command,
    parse_identity_text,
    read_pending_input,
)
from app.i18n import Locale
from app.models.profile import PersonProfile
from app.models.state import IdentityQuery, PeopleSearchState

log = structlog.get_logger()

_APPROVE_WORDS = {"yes", "y", "да", "д", "ok", "ок", "approve"}
_REJECT_WORDS = {"no", "n", "нет", "н"}
_ABORT_WORDS = {"abort", "cancel", "stop"}


# ----- pure message helpers -----


def _text_of(message: Message) -> str:
    out: list[str] = []
    for part in message.parts or []:
        root = part.root
        if isinstance(root, TextPart) and root.text:
            out.append(root.text)
    return " ".join(out).strip()


def _data_of(message: Message) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in message.parts or []:
        root = part.root
        if isinstance(root, DataPart) and isinstance(root.data, dict):
            merged.update(root.data)
    return merged


def locale_from_message(message: Message) -> Locale:
    meta = message.metadata or {}
    loc = meta.get("locale")
    return cast(Locale, loc) if loc in ("en", "ru") else "en"


def message_to_identity(message: Message) -> dict[str, Any]:
    data = _data_of(message)
    if data.get("first_name"):
        return {k: v for k, v in data.items() if v}
    return parse_identity_text(_text_of(message))


def message_to_answer(message: Message, pending: PendingInput) -> ResumeAnswer:
    data = _data_of(message)
    text = _text_of(message)
    if pending.kind == "ask_identity":
        if data.get("first_name"):
            return ResumeAnswer(identity={k: v for k, v in data.items() if v})
        return ResumeAnswer(identity=parse_identity_text(text))
    if pending.kind == "ask_narrowing":
        if isinstance(data.get("pick_index"), int):
            return ResumeAnswer(pick_index=data["pick_index"])
        if data.get("attribute") and data.get("value"):
            return ResumeAnswer(attribute_value=(data["attribute"], data["value"]))
        if pending.attribute and text:
            return ResumeAnswer(attribute_value=(pending.attribute, text))
        return ResumeAnswer(extra=text)
    # confirm_profile
    if isinstance(data.get("decision"), str):
        return ResumeAnswer(decision=data["decision"])
    low = text.lower().strip()
    if low in _ABORT_WORDS:
        return ResumeAnswer(decision="abort")
    if low in _REJECT_WORDS:
        return ResumeAnswer(decision="more")
    if low in _APPROVE_WORDS:
        return ResumeAnswer(decision="approve")
    return ResumeAnswer(decision="more", extra=text)


def pending_to_parts(pending: PendingInput) -> list[Part]:
    parts: list[Part] = [Part(root=TextPart(text=pending.question or ""))]
    if pending.data:
        parts.append(Part(root=DataPart(data=pending.data)))
    return parts


def profile_to_artifact_parts(profile: dict[str, Any]) -> list[Part]:
    return [Part(root=DataPart(data=profile))]


# ----- executor -----


class PeopleSearchExecutor(AgentExecutor):
    """Drives the compiled people-search graph for one A2A task."""

    def __init__(self, graph: Any, current_user_id: Any) -> None:
        # `graph` is the compiled LangGraph; `current_user_id` is a zero-arg
        # callable returning the authenticated user_id (the auth contextvar
        # getter) so the executor stays decoupled from the auth module.
        self._graph = graph
        self._current_user_id = current_user_id

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        message: Message = context.message  # type: ignore[assignment]
        task = context.current_task
        if task is None:
            task = new_task(message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        thread_id = task.id
        config = {"configurable": {"thread_id": thread_id}}
        locale = locale_from_message(message)

        snapshot = await self._graph.aget_state(config)
        pending = read_pending_input(snapshot, locale)
        if pending is not None:
            answer = message_to_answer(message, pending)
            graph_input: Any = build_resume_command(pending, answer)
        else:
            query = cast(IdentityQuery, message_to_identity(message))
            graph_input = PeopleSearchState(query=query, locale=locale)

        try:
            async for _event in self._graph.astream(
                graph_input, config=config, stream_mode="updates"
            ):
                await updater.update_status(TaskState.working)
        except Exception as exc:
            log.warning("a2a_graph_failed", error=str(exc))
            await updater.failed(
                message=updater.new_agent_message(
                    [Part(root=TextPart(text=f"Search failed: {exc}"))]
                )
            )
            return

        snapshot = await self._graph.aget_state(config)
        pending = read_pending_input(snapshot, locale)
        if pending is not None:
            await updater.requires_input(
                message=updater.new_agent_message(pending_to_parts(pending))
            )
            return

        state = dict(snapshot.values) if snapshot else {}
        profile = state.get("profile")
        if state.get("phase") == "done" and state.get("user_decision") != "abort" and profile:
            await self._persist(thread_id, profile)
            await updater.add_artifact(
                profile_to_artifact_parts(profile), name="person_profile"
            )
            await updater.complete()
        else:
            await updater.failed(
                message=updater.new_agent_message(
                    [Part(root=TextPart(text="No profile could be built for this query."))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            return
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.cancel()

    async def _persist(self, thread_id: str, profile_raw: dict[str, Any]) -> None:
        user_id = self._current_user_id()
        if not user_id:
            return
        try:
            profile = PersonProfile.model_validate(profile_raw)
        except Exception:
            return
        sources = [
            {"url": str(ev.url), "platform": ev.platform, "snippet": ev.snippet}
            for ev in profile.evidence
        ]
        try:
            await save_profile(
                user_id=int(user_id),
                thread_id=thread_id,
                full_name=profile.full_name,
                profile=profile.model_dump(mode="json"),
                sources=sources,
            )
        except Exception as exc:
            log.warning("a2a_persist_failed", error=str(exc))
