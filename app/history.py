"""Pure chat-history session logic — no Chainlit, no DB.

These helpers back the persistent-history feature in `app/main.py`:

- `build_fresh_input` — the graph input for *starting* a search, with all
  transient search state cleared so a follow-up on an already-finished thread
  does not inherit a stale iteration count / candidate list / profile.
- `derive_awaiting` / `resume_awaiting` — recompute the UI's `awaiting` flag from
  a LangGraph state snapshot. The checkpoint is the source of truth for "what is
  this conversation waiting for", so resume restores the flag without re-rendering.
- `compute_thread_tags` / `render_tag_line` — namespaced auto-tags for a finished
  search and a compact, searchable line that makes them discoverable via the
  Chainlit sidebar search box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.graph.bridge import read_pending_input
from app.i18n import Locale, t
from app.models.profile import PersonProfile
from app.models.state import IdentityQuery, PeopleSearchState

# Maps a paused interrupt kind onto the UI's `awaiting` token.
_AWAITING_BY_KIND: dict[str, str] = {
    "ask_identity": "identity",
    "ask_narrowing": "narrowing",
    "confirm_profile": "confirm",
}


@dataclass(slots=True)
class AwaitingState:
    """The server-side session flags needed to route the next user reply."""

    awaiting: str | None
    narrowing_attribute: str | None = None
    narrowing_candidate_count: int = 0


def build_fresh_input(query: IdentityQuery, locale: Locale) -> PeopleSearchState:
    """Graph input that starts a fresh search, clearing transient state.

    Used both for a brand-new thread (channels already empty — the resets are
    no-ops) and for a follow-up search on a finished thread, where the previous
    run left `iteration`, `candidates`, `profile`, … in the checkpoint.

    Note: `max_iterations` is intentionally not reset here; it is re-established
    from `Settings` by the `collect_identity` node, so carrying the config default
    is correct.
    """
    return PeopleSearchState(
        query=query,
        locale=locale,
        candidates=[],
        visited_urls=[],
        fetched_pages=[],
        iteration=0,
        phase="collect",
        profile=None,
        user_decision=None,
        selected_candidate_index=None,
        guard_block=None,
    )


def derive_awaiting(snapshot: Any, locale: Locale) -> AwaitingState:
    """Recompute the `awaiting` flag from a graph snapshot.

    Returns `awaiting=None` when the graph is not paused on an interrupt (the
    live message path treats that as "the run finished").
    """
    pending = read_pending_input(snapshot, locale)
    if pending is None:
        return AwaitingState(awaiting=None)
    return AwaitingState(
        awaiting=_AWAITING_BY_KIND.get(pending.kind),
        narrowing_attribute=pending.attribute,
        narrowing_candidate_count=pending.candidate_count,
    )


def resume_awaiting(snapshot: Any, locale: Locale) -> AwaitingState:
    """Like `derive_awaiting`, but for resuming a thread from the sidebar.

    A thread with no active interrupt is either finished or has a lost
    checkpoint; either way we fall back to `identity` so the user can start a
    fresh search in that same thread rather than hitting a dead end.
    """
    state = derive_awaiting(snapshot, locale)
    if state.awaiting is None:
        return AwaitingState(awaiting="identity")
    return state


def compute_thread_tags(profile: PersonProfile, locale: str) -> list[str]:
    """Namespaced, deduped, deterministic tags for a finished search.

    `platform:<name>` for each distinct evidence platform (sorted), then
    `confidence:<level>` and `locale:<loc>`. Tokens stay untranslated so they are
    stable identifiers across languages.
    """
    platforms = sorted(
        {p for ev in profile.evidence if (p := (ev.platform or "").strip().lower())}
    )
    tags = [f"platform:{p}" for p in platforms]
    tags.append(f"confidence:{profile.confidence}")
    tags.append(f"locale:{locale}")
    return tags


def render_tag_line(tags: list[str], locale: Locale) -> str:
    """A compact, searchable one-liner echoing the tags into thread content."""
    return f"🏷 {t('history_tagged', locale)} " + " · ".join(tags)
