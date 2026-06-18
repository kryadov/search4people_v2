"""Chainlit entrypoint.

Wires the password auth callback, per-session graph instantiation, language
toggle (`/ru`, `/en`), and streaming of LangGraph events as `cl.Step`s.

Usage:
    uv run chainlit run app/main.py --port 8000
"""

from __future__ import annotations

import contextlib
import re
from typing import Any, cast

import chainlit as cl
import chainlit.data as cl_data
import structlog
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.auth import password_auth  # noqa: F401 - registers @cl.password_auth_callback
from app.config import get_settings
from app.db.chat_history import build_data_layer, init_chat_history_db
from app.db.connection import init_db
from app.db.profiles import save_profile
from app.db.users import set_user_locale
from app.graph.bridge import (
    PendingInput,
    ResumeAnswer,
    build_resume_command,
    fresh_search_input,
    parse_identity_text,
    read_pending_input,
)
from app.graph.build import build_graph
from app.i18n import DEFAULT_LOCALE, Locale, detect_locale_command, t
from app.models.profile import PersonProfile
from app.models.state import IdentityQuery, PeopleSearchState
from app.ui.profile_card import render_profile_message
from app.ui.session_state import derive_session_state
from app.ui.steps import emit_node_step
from app.ui.tags import compute_auto_tags, render_tag_line

log = structlog.get_logger()


@cl.data_layer
def _build_chat_history_layer() -> SQLAlchemyDataLayer:
    """Register Chainlit's persistent chat-history store (per-user threads)."""
    return build_data_layer()


# ----- Process-wide checkpointer -----
# AsyncSqliteSaver is a context manager. We enter it once at server start and
# exit on shutdown, then reuse the resulting saver across every chat session.

_saver_ctx: Any = None
_saver: AsyncSqliteSaver | None = None
_graph = None


async def _ensure_graph():
    global _saver_ctx, _saver, _graph
    if _graph is not None:
        return _graph
    settings = get_settings()
    await init_db()
    await init_chat_history_db()
    _saver_ctx = AsyncSqliteSaver.from_conn_string(str(settings.db_path))
    _saver = await _saver_ctx.__aenter__()
    _graph = build_graph().compile(checkpointer=_saver)
    log.info("graph_compiled", db=str(settings.db_path))
    return _graph


@cl.on_app_shutdown
async def _shutdown() -> None:
    global _saver_ctx, _saver, _graph
    if _saver_ctx is not None:
        with contextlib.suppress(Exception):
            await _saver_ctx.__aexit__(None, None, None)
    _saver = None
    _saver_ctx = None
    _graph = None


# ----- Helpers -----


def _user_locale() -> Locale:
    user = cl.user_session.get("user")
    if user and user.metadata and user.metadata.get("locale") in ("en", "ru"):
        return user.metadata["locale"]
    stored = cl.user_session.get("locale")
    if stored in ("en", "ru"):
        return stored
    return DEFAULT_LOCALE


async def _set_locale(locale: Locale) -> None:
    cl.user_session.set("locale", locale)
    user = cl.user_session.get("user")
    if user and user.metadata and "id" in user.metadata:
        try:
            await set_user_locale(int(user.metadata["id"]), locale)
        except Exception as exc:
            log.warning("locale_persist_failed", error=str(exc))


def _thread_config() -> dict[str, Any]:
    # The graph thread id IS Chainlit's persisted thread id, so resuming a
    # thread restores both the UI messages and the LangGraph checkpoint.
    return {"configurable": {"thread_id": cl.context.session.thread_id}}


def _format_candidate_list(candidates: list[dict[str, Any]], limit: int = 8) -> str:
    rows: list[str] = []
    for i, c in enumerate(candidates[:limit], start=1):
        platform = c.get("platform") or "web"
        title = c.get("title") or "(no title)"
        url = c.get("url") or ""
        snippet = (c.get("snippet") or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "…"
        rows.append(f"**{i}.** [{platform}] [{title}]({url})\n   {snippet}".rstrip())
    return "\n\n".join(rows)


_PICK_INDEX_RE = re.compile(r"^\s*#?\s*(\d{1,2})\s*$")


def _parse_pick_index(text: str, count: int) -> int | None:
    """Return a 0-based index if the user typed `#N` / `N`, else None."""
    m = _PICK_INDEX_RE.match(text)
    if not m:
        return None
    n = int(m.group(1))
    if 1 <= n <= count:
        return n - 1
    return None


async def _stream_graph(initial_input: Any) -> dict[str, Any]:
    """Run the graph, emit per-node steps, return the final state snapshot."""
    graph = await _ensure_graph()
    config = _thread_config()
    locale = _user_locale()
    final_state: dict[str, Any] = {}
    async for event in graph.astream(initial_input, config=config, stream_mode="updates"):
        # `event` looks like {"node_name": {...patch...}}; emit a step per node.
        for node, patch in event.items():
            if node == "__interrupt__":
                continue
            await emit_node_step(node, locale, patch)
            if isinstance(patch, dict):
                final_state.update(patch)
    # Re-read the full snapshot from the checkpointer so we have everything.
    snapshot = await graph.aget_state(config)
    if snapshot:
        return dict(snapshot.values)
    return final_state


async def _handle_interrupt_and_render(state: dict[str, Any]) -> bool:
    """If the graph is paused on an interrupt, render the matching Chainlit
    prompt and return True (so the next user reply is treated as a resume)."""
    graph = await _ensure_graph()
    config = _thread_config()
    snapshot = await graph.aget_state(config)
    locale = _user_locale()
    pending = read_pending_input(snapshot, locale)
    if pending is None:
        return False

    if pending.kind == "ask_identity":
        await cl.Message(content=pending.question).send()
        cl.user_session.set("awaiting", "identity")
        return True

    if pending.kind == "ask_narrowing":
        candidates = pending.data.get("candidates") or []
        options = pending.data.get("options") or []
        attribute = pending.attribute
        parts: list[str] = []
        if candidates:
            parts.append(t("candidates_heading", locale, count=len(candidates)))
            parts.append(_format_candidate_list(candidates))
        prefix = t("ask_more_details_prefix", locale)
        parts.append(f"{prefix} {pending.question or ''}".strip())
        if options and attribute:
            parts.append(t("options_heading", locale, attribute=attribute))
            parts.append("\n".join(f"- {o}" for o in options))
        parts.append(t("narrowing_reply_hint", locale))
        await cl.Message(content="\n\n".join(parts)).send()
        cl.user_session.set("awaiting", "narrowing")
        cl.user_session.set("narrowing_attribute", attribute)
        cl.user_session.set("narrowing_candidate_count", len(candidates))
        return True

    if pending.kind == "confirm_profile":
        profile_raw = pending.data.get("profile")
        if profile_raw:
            profile = PersonProfile.model_validate(profile_raw)
            await render_profile_message(profile, locale).send()
        await cl.Message(content=pending.question).send()
        cl.user_session.set("awaiting", "confirm")
        return True

    return False


async def _persist_if_done(state: dict[str, Any]) -> None:
    """Save the profile to SQLite once the graph reaches its final state."""
    profile_raw = state.get("profile")
    if not profile_raw:
        return
    user = cl.user_session.get("user")
    if not user:
        return
    user_id = int((user.metadata or {}).get("id") or 0)
    if not user_id:
        return
    try:
        profile = PersonProfile.model_validate(profile_raw)
    except Exception:
        return
    thread_id = cl.context.session.thread_id
    sources = [
        {"url": str(ev.url), "platform": ev.platform, "snippet": ev.snippet}
        for ev in profile.evidence
    ]
    try:
        await save_profile(
            user_id=user_id,
            thread_id=str(thread_id),
            full_name=profile.full_name,
            profile=profile.model_dump(mode="json"),
            sources=sources,
        )
        await cl.Message(content=t("profile_saved", _user_locale())).send()
    except Exception as exc:
        log.warning("persist_profile_failed", error=str(exc))
        return

    # Auto-tags: discoverable via the built-in search (echo line) and stored in
    # thread metadata for future faceting. NOTE: we use metadata, not the
    # data layer's `tags=` param, because SQLite cannot bind a Python list.
    locale = _user_locale()
    tags = compute_auto_tags(profile.model_dump(mode="json"), locale)
    data_layer = cl_data.get_data_layer()
    if data_layer is not None:
        try:
            await data_layer.update_thread(str(thread_id), metadata={"auto_tags": tags})
        except Exception as exc:
            log.warning("thread_tags_failed", error=str(exc))
    await cl.Message(content=render_tag_line(tags, locale)).send()


# ----- Chainlit lifecycle -----


@cl.on_chat_start
async def on_chat_start() -> None:
    await _ensure_graph()
    locale = _user_locale()
    await cl.Message(content=t("disclaimer", locale)).send()
    await cl.Message(content=t("language_toggle_hint", locale)).send()
    await cl.Message(content=t("ask_who", locale)).send()
    cl.user_session.set("awaiting", "identity")


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """Restore server-side session flags when a past thread is reopened.

    Chainlit replays the stored messages itself; we only need to recover
    `awaiting`/locale. The graph checkpoint is the source of truth for what the
    conversation is waiting for, so we derive it from the snapshot.
    """
    graph = await _ensure_graph()
    config = _thread_config()
    locale = _user_locale()
    snapshot = await graph.aget_state(config)
    pending = read_pending_input(snapshot, locale)
    sess = derive_session_state(pending)
    cl.user_session.set("awaiting", sess.awaiting)
    if sess.awaiting == "narrowing":
        cl.user_session.set("narrowing_attribute", sess.narrowing_attribute)
        cl.user_session.set("narrowing_candidate_count", sess.narrowing_candidate_count)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    text = (message.content or "").strip()

    # Language toggle takes precedence over everything else.
    detected = detect_locale_command(text)
    if detected:
        await _set_locale(detected)
        await cl.Message(content=t("language_set", detected)).send()
        return

    awaiting = cl.user_session.get("awaiting")
    graph_input: Any

    if awaiting == "identity":
        parsed = parse_identity_text(text)
        if "first_name" not in parsed or "last_name" not in parsed:
            await cl.Message(content=t("ask_who", _user_locale())).send()
            return
        # Has the graph already been started in this thread? If so resume; else fresh.
        config = _thread_config()
        graph = await _ensure_graph()
        snapshot = await graph.aget_state(config)
        pending = read_pending_input(snapshot, _user_locale())
        if pending is not None:
            graph_input = build_resume_command(pending, ResumeAnswer(identity=parsed))
        else:
            graph_input = PeopleSearchState(query=cast(IdentityQuery, parsed), locale=_user_locale())
    elif awaiting == "narrowing":
        attribute = cl.user_session.get("narrowing_attribute")
        cand_count = int(cl.user_session.get("narrowing_candidate_count") or 0)
        picked = _parse_pick_index(text, cand_count) if cand_count else None
        pending = PendingInput(
            kind="ask_narrowing", question="", data={}, candidate_count=cand_count, attribute=attribute
        )
        if picked is not None:
            answer = ResumeAnswer(pick_index=picked)
        elif attribute:
            answer = ResumeAnswer(attribute_value=(attribute, text))
        else:
            answer = ResumeAnswer(extra=text)
        graph_input = build_resume_command(pending, answer)
    elif awaiting == "confirm":
        normalized = text.lower()
        if normalized in {"yes", "y", "да", "д", "ok", "ок"}:
            answer = ResumeAnswer(decision="approve")
        elif normalized in {"no", "n", "нет", "н"}:
            answer = ResumeAnswer(decision="more")
        else:
            answer = ResumeAnswer(decision="more", extra=text)
        pending = PendingInput(
            kind="confirm_profile", question="", data={}, candidate_count=0, attribute=None
        )
        graph_input = build_resume_command(pending, answer)
    else:
        # No active expectation (e.g. a finished thread) — start a new search in
        # the SAME thread, resetting transient state but keeping the history.
        parsed = parse_identity_text(text)
        graph_input = cast(PeopleSearchState, fresh_search_input(parsed, _user_locale()))

    cl.user_session.set("awaiting", None)
    state = await _stream_graph(graph_input)

    # If the graph paused on a new interrupt, render its prompt.
    if await _handle_interrupt_and_render(state):
        return

    # A guardrail blocked the request — show the refusal and stop.
    if state.get("guard_block"):
        await cl.Message(content=t("guard_blocked", _user_locale())).send()
        return

    # Otherwise we're done — persist + report.
    if state.get("phase") == "done" and state.get("user_decision") != "abort":
        await _persist_if_done(state)
    elif not state.get("profile"):
        await cl.Message(content=t("not_found", _user_locale())).send()
