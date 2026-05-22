"""Chainlit entrypoint.

Wires the password auth callback, per-session graph instantiation, language
toggle (`/ru`, `/en`), and streaming of LangGraph events as `cl.Step`s.

Usage:
    uv run chainlit run app/main.py --port 8000
"""

from __future__ import annotations

import contextlib
import re
import uuid
from typing import Any

import chainlit as cl
import structlog
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.auth import password_auth  # noqa: F401 - registers @cl.password_auth_callback
from app.config import get_settings
from app.db.connection import init_db
from app.db.profiles import save_profile
from app.db.users import set_user_locale
from app.graph.build import build_graph
from app.i18n import DEFAULT_LOCALE, Locale, detect_locale_command, t
from app.models.profile import PersonProfile
from app.models.state import IdentityQuery, PeopleSearchState
from app.ui.profile_card import render_profile_message
from app.ui.steps import emit_node_step

log = structlog.get_logger()


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
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", thread_id)
    return {"configurable": {"thread_id": thread_id}}


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


def _parse_identity(message: str) -> IdentityQuery:
    """Very small heuristic: first token = first_name, rest = last_name."""
    tokens = [t for t in message.strip().split() if t]
    if not tokens:
        return IdentityQuery()
    if len(tokens) == 1:
        return IdentityQuery(first_name=tokens[0])
    return IdentityQuery(first_name=tokens[0], last_name=" ".join(tokens[1:]))


async def _stream_graph(initial_input: PeopleSearchState | Command) -> dict[str, Any]:
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
    """Look at the current snapshot. If the graph is paused on an interrupt,
    render the matching prompt to the user and return True (so on_message
    handles the next user reply as a Resume).
    """
    graph = await _ensure_graph()
    config = _thread_config()
    snapshot = await graph.aget_state(config)
    interrupts = snapshot.tasks and any(t.interrupts for t in snapshot.tasks)
    if not interrupts:
        return False

    locale = _user_locale()
    for task in snapshot.tasks:
        for itr in task.interrupts:
            payload = itr.value if hasattr(itr, "value") else itr
            kind = (payload or {}).get("kind") if isinstance(payload, dict) else None
            if kind == "ask_identity":
                await cl.Message(content=t("ask_who", locale)).send()
                cl.user_session.set("awaiting", "identity")
                return True
            if kind == "ask_narrowing":
                pd = payload if isinstance(payload, dict) else {}
                question = pd.get("question")
                attribute = pd.get("attribute")
                candidates = pd.get("candidates") or []
                options = pd.get("options") or []

                parts: list[str] = []
                if candidates:
                    parts.append(t("candidates_heading", locale, count=len(candidates)))
                    parts.append(_format_candidate_list(candidates))
                prefix = t("ask_more_details_prefix", locale)
                parts.append(f"{prefix} {question or ''}".strip())
                if options and attribute:
                    parts.append(t("options_heading", locale, attribute=attribute))
                    parts.append("\n".join(f"- {o}" for o in options))
                parts.append(t("narrowing_reply_hint", locale))

                await cl.Message(content="\n\n".join(parts)).send()
                cl.user_session.set("awaiting", "narrowing")
                cl.user_session.set("narrowing_attribute", attribute)
                cl.user_session.set("narrowing_candidate_count", len(candidates))
                return True
            if kind == "confirm_profile":
                profile_raw = (payload or {}).get("profile") if isinstance(payload, dict) else None
                if profile_raw:
                    profile = PersonProfile.model_validate(profile_raw)
                    await render_profile_message(profile, locale).send()
                await cl.Message(content=t("confirm_profile", locale)).send()
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
    thread_id = cl.user_session.get("thread_id") or ""
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


# ----- Chainlit lifecycle -----


@cl.on_chat_start
async def on_chat_start() -> None:
    await _ensure_graph()
    locale = _user_locale()
    await cl.Message(content=t("disclaimer", locale)).send()
    await cl.Message(content=t("language_toggle_hint", locale)).send()
    await cl.Message(content=t("ask_who", locale)).send()
    cl.user_session.set("awaiting", "identity")


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
    graph_input: PeopleSearchState | Command

    if awaiting == "identity":
        parsed = _parse_identity(text)
        if "first_name" not in parsed or "last_name" not in parsed:
            await cl.Message(content=t("ask_who", _user_locale())).send()
            return
        # Has the graph already been started in this thread? If so resume; else fresh.
        config = _thread_config()
        graph = await _ensure_graph()
        snapshot = await graph.aget_state(config)
        if snapshot and snapshot.tasks and any(t.interrupts for t in snapshot.tasks):
            graph_input = Command(resume=parsed)
        else:
            graph_input = PeopleSearchState(query=parsed, locale=_user_locale())
    elif awaiting == "narrowing":
        attribute = cl.user_session.get("narrowing_attribute")
        cand_count = int(cl.user_session.get("narrowing_candidate_count") or 0)
        picked = _parse_pick_index(text, cand_count) if cand_count else None
        if picked is not None:
            graph_input = Command(resume={"pick_index": picked})
        elif attribute:
            graph_input = Command(resume={"attribute": attribute, "value": text})
        else:
            graph_input = Command(resume={"extra": text})
    elif awaiting == "confirm":
        normalized = text.lower()
        if normalized in {"yes", "y", "да", "д", "ok", "ок"}:
            decision: dict[str, Any] = {"decision": "approve"}
        elif normalized in {"no", "n", "нет", "н"}:
            decision = {"decision": "more"}
        else:
            decision = {"decision": "more", "extra": text}
        graph_input = Command(resume=decision)
    else:
        # No active expectation — treat as a fresh start.
        cl.user_session.set("thread_id", str(uuid.uuid4()))
        parsed = _parse_identity(text)
        graph_input = PeopleSearchState(query=parsed, locale=_user_locale())

    cl.user_session.set("awaiting", None)
    state = await _stream_graph(graph_input)

    # If the graph paused on a new interrupt, render its prompt.
    if await _handle_interrupt_and_render(state):
        return

    # Otherwise we're done — persist + report.
    if state.get("phase") == "done" and state.get("user_decision") != "abort":
        await _persist_if_done(state)
    elif not state.get("profile"):
        await cl.Message(content=t("not_found", _user_locale())).send()
