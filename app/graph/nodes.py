"""Graph nodes.

Each node receives the current `PeopleSearchState` and returns a partial state
patch. Where we need human input we call `langgraph.types.interrupt(...)`,
which suspends the graph until the Chainlit layer resumes it with a `Command`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from app.config import get_settings
from app.graph.prompts import (
    MERGE_PROFILE_PROMPT,
    NARROW_QUERY_PROMPT,
    SYSTEM_RESEARCHER,
)
from app.guardrails import get_guardrails
from app.llm import build_chat_model, build_structured_model
from app.models.profile import Candidate, Evidence, PersonProfile
from app.models.state import IdentityQuery, PeopleSearchState
from app.tools.extract import extract_profile_from_page
from app.tools.fetch import PageFetcher
from app.tools.search import search_many_platforms

log = structlog.get_logger()

# ----- Module-scoped shared resources -----
# A single PageFetcher per process keeps an HTTPX client warm and reuses one
# Playwright browser invocation across calls.
_fetcher: PageFetcher | None = None


def _get_fetcher() -> PageFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = PageFetcher()
    return _fetcher


# ----- Helpers -----


def _distinguishers(query: IdentityQuery) -> str:
    bits: list[str] = []
    for key, label in [
        ("age", "age"),
        ("country", "country"),
        ("city", "city"),
        ("school", "school"),
        ("university", "university"),
        ("employer", "employer"),
        ("extra", "other"),
    ]:
        value = query.get(key)
        if value:
            bits.append(f"{label}={value}")
    return ", ".join(bits)


def _full_name(query: IdentityQuery) -> str:
    return " ".join(filter(None, [query.get("first_name"), query.get("last_name")]))


def _query_string(query: IdentityQuery) -> str:
    name = _full_name(query)
    parts = [f'"{name}"' if name else ""]
    for key in ("city", "country", "employer", "university", "school"):
        v = query.get(key)
        if v:
            parts.append(f'"{v}"')
    return " ".join(p for p in parts if p)


def _short_candidate_list(candidates: list[dict[str, Any]], limit: int = 8) -> str:
    rows: list[str] = []
    for i, c in enumerate(candidates[:limit], start=1):
        rows.append(
            f"{i}. [{c.get('platform') or 'web'}] {c.get('title') or '(no title)'} — {c.get('url')}\n"
            f"     {c.get('snippet') or ''}"
        )
    return "\n".join(rows)


def _coerce_candidates(hits: list) -> list[dict[str, Any]]:
    """Convert raw SearchHits into validated, serializable Candidate dicts."""
    out: list[dict[str, Any]] = []
    for h in hits:
        if not h.url:
            continue
        try:
            cand = Candidate(
                url=h.url,
                title=h.title,
                snippet=h.snippet,
                platform=h.platform,
            )
        except Exception as exc:
            log.debug("candidate_skipped", url=h.url, error=str(exc))
            continue
        out.append(cand.model_dump(mode="json"))
    return out


# ----- Node implementations -----


async def collect_identity(state: PeopleSearchState) -> dict[str, Any]:
    """Make sure we have at least first_name + last_name. If not, interrupt."""
    query = dict(state.get("query") or {})
    if not query.get("first_name") or not query.get("last_name"):
        payload = interrupt(
            {
                "kind": "ask_identity",
                "locale": state.get("locale", "en"),
            }
        )
        # Resume value is a dict like {"first_name": "...", "last_name": "..."}.
        if isinstance(payload, dict):
            query.update({k: v for k, v in payload.items() if v})
    full_name = " ".join(str(p) for p in (query.get("first_name"), query.get("last_name")) if p)
    if full_name:
        verdict = await get_guardrails().check_input(full_name)
        if verdict.blocked:
            return {"guard_block": {"reason": verdict.reason, "point": "input"}, "phase": "done"}
    settings = get_settings()
    return {
        "query": query,
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", settings.max_iterations),
        "phase": "preliminary",
        "candidates": state.get("candidates", []),
        "visited_urls": state.get("visited_urls", []),
        "fetched_pages": state.get("fetched_pages", []),
    }


def route_after_collect(state: PeopleSearchState) -> str:
    """Stop early if a guardrail blocked the identity query; else search."""
    if state.get("guard_block"):
        return "blocked"
    return "preliminary_search"


async def preliminary_search(state: PeopleSearchState) -> dict[str, Any]:
    """Run the top-3 platform search in parallel."""
    settings = get_settings()
    query = state["query"]
    q = _query_string(query)
    hits = await search_many_platforms(settings.platforms_primary, q, limit_per_platform=6)
    candidates = _coerce_candidates(hits)
    return {
        "candidates": candidates,
        "phase": "evaluate",
        "iteration": state.get("iteration", 0) + 1,
    }


async def evaluate_candidates(state: PeopleSearchState) -> dict[str, Any]:
    """Routing node — sets the next phase but doesn't change candidates."""
    count = len(state.get("candidates") or [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", get_settings().max_iterations)

    if count == 0 and iteration < max_iter:
        return {"phase": "expand"}
    if count > 1 and iteration < max_iter:
        return {"phase": "narrow"}
    # Either we have exactly one, or we hit the iteration ceiling.
    return {"phase": "fetch"}


def route_after_evaluate(state: PeopleSearchState) -> str:
    phase = state.get("phase", "fetch")
    if phase == "narrow":
        return "narrow_query"
    if phase == "expand":
        return "expand_search"
    return "fetch_pages"


async def plan_narrowing(
    candidates: list[dict[str, Any]], query: IdentityQuery, locale: str
) -> dict[str, Any]:
    """Ask the LLM which distinguishing attribute to request next.

    Pure (no `interrupt`) so it can be evaluated in isolation. Returns a dict
    with keys: `attribute` (str | None), `question` (str), `options` (list[str]).
    """
    prompt = NARROW_QUERY_PROMPT.format(
        candidates=_short_candidate_list(candidates),
        known_attributes=_distinguishers(query) or "(none yet)",
    )
    model = build_chat_model(temperature=0.0)
    response = await model.ainvoke(
        [SystemMessage(content=SYSTEM_RESEARCHER), HumanMessage(content=prompt)]
    )
    plan = _safe_json(response.content) or {}
    attribute = plan.get("attribute")
    question = (
        plan.get(f"question_{locale}")
        or "Could you share any additional distinguishing details?"
    )
    raw_options = plan.get("options") or []
    options: list[str] = []
    if isinstance(raw_options, list):
        seen: set[str] = set()
        for o in raw_options:
            if isinstance(o, str):
                stripped = o.strip()
                if stripped and stripped.lower() not in seen:
                    options.append(stripped)
                    seen.add(stripped.lower())
    return {"attribute": attribute, "question": question, "options": options}


async def narrow_query(state: PeopleSearchState) -> dict[str, Any]:
    """Ask the LLM which attribute to ask for, then interrupt for the user.

    The user can either pick a candidate by index (→ jump straight to fetch),
    pick one of the LLM-extracted attribute values, or type a free-form value.
    """
    candidates = state.get("candidates") or []
    query = state["query"]
    locale = state.get("locale", "en")

    plan = await plan_narrowing(candidates, query, locale)
    attribute = plan["attribute"]
    question = plan["question"]
    options = plan["options"]

    answer = interrupt(
        {
            "kind": "ask_narrowing",
            "attribute": attribute,
            "question": question,
            "options": options,
            # Surface the candidate list so the UI can render it for selection.
            "candidates": list(candidates),
            "locale": locale,
        }
    )

    # User picked a specific candidate by index → skip further narrowing and
    # jump straight to fetch_pages with that selection.
    if isinstance(answer, dict) and isinstance(answer.get("pick_index"), int):
        idx = answer["pick_index"]
        if 0 <= idx < len(candidates):
            return {
                "selected_candidate_index": idx,
                "phase": "fetch",
            }

    new_query = dict(query)
    if isinstance(answer, dict):
        # UI sends {"attribute": "...", "value": "..."} OR {"extra": "..."}.
        if attr := answer.get("attribute"):
            value = answer.get("value") or answer.get(attr)
            if value:
                new_query[attr] = value
        elif text := answer.get("extra") or answer.get("text"):
            new_query["extra"] = text
    elif isinstance(answer, str):
        new_query["extra"] = answer

    probe = " ".join(str(v) for v in new_query.values() if v)
    verdict = await get_guardrails().check_input(probe)
    if verdict.blocked:
        return {"guard_block": {"reason": verdict.reason, "point": "input"}, "phase": "done"}

    return {"query": new_query, "phase": "preliminary"}


def route_after_narrow(state: PeopleSearchState) -> str:
    """Either jump straight to fetch (user picked a candidate) or re-search."""
    if state.get("guard_block"):
        return "blocked"
    if state.get("phase") == "fetch":
        return "fetch_pages"
    return "preliminary_search"


async def expand_search(state: PeopleSearchState) -> dict[str, Any]:
    """Search the secondary platform list + a general web sweep."""
    settings = get_settings()
    query = state["query"]
    q = _query_string(query)
    secondary_hits = await search_many_platforms(
        settings.platforms_secondary, q, limit_per_platform=5
    )
    from app.tools.search import web_search

    general_hits = await web_search(q, limit=10)
    candidates = _coerce_candidates([*secondary_hits, *general_hits])
    # Merge with anything we already had.
    existing = list(state.get("candidates") or [])
    seen = {c.get("url") for c in existing}
    for c in candidates:
        if c.get("url") not in seen:
            existing.append(c)
            seen.add(c.get("url"))
    return {
        "candidates": existing,
        "phase": "evaluate",
        "iteration": state.get("iteration", 0) + 1,
    }


async def fetch_pages(state: PeopleSearchState) -> dict[str, Any]:
    """Fetch candidate URLs in parallel (bounded) and extract per-page profiles."""
    candidates = state.get("candidates") or []
    visited = set(state.get("visited_urls") or [])
    fetched = list(state.get("fetched_pages") or [])
    full_name = _full_name(state["query"])
    distinguishers = _distinguishers(state["query"])

    selected_index = state.get("selected_candidate_index")
    if selected_index is not None and 0 <= selected_index < len(candidates):
        targets = [candidates[selected_index]]
    else:
        # Cap fan-out to keep latency + cost predictable.
        targets = [c for c in candidates if c.get("url") not in visited][:6]

    fetcher = _get_fetcher()
    sem = asyncio.Semaphore(3)

    async def _fetch_and_extract(c: dict[str, Any]) -> dict[str, Any] | None:
        url = c.get("url")
        if not url:
            return None
        title = c.get("title")
        snippet = c.get("snippet")
        async with sem:
            result = await fetcher.fetch(url)
            markdown = result.markdown or ""
            # Untrusted page content is a prompt-injection vector before it
            # reaches the extraction LLM; sanitize anything the guard flags.
            if markdown:
                scan = await get_guardrails().scan_content(markdown)
                if scan.transformed_text is not None:
                    markdown = scan.transformed_text
            # Bail only when there is nothing at all to extract from: a blocked
            # page (empty body) still yields facts from the search snippet/title.
            if not markdown and not (snippet or title):
                return None
            extracted = await extract_profile_from_page(
                full_name=full_name,
                distinguishers=distinguishers,
                url=url,
                markdown=markdown,
                platform=c.get("platform"),
                title=title,
                snippet=snippet,
            )
            return {
                "url": url,
                "platform": c.get("platform"),
                "snippet": snippet,
                "markdown_len": len(markdown),
                "fetch_blocked": not markdown,
                "partial": extracted.model_dump(mode="json"),
            }

    results = await asyncio.gather(*(_fetch_and_extract(c) for c in targets))
    for r in results:
        if r is None:
            continue
        fetched.append(r)
        visited.add(r["url"])

    return {
        "fetched_pages": fetched,
        "visited_urls": list(visited),
        "phase": "build",
        "selected_candidate_index": None,
    }


async def build_profile(state: PeopleSearchState) -> dict[str, Any]:
    """Merge per-page partials into a single PersonProfile via the LLM."""
    full_name = _full_name(state["query"])
    distinguishers = _distinguishers(state["query"])
    fetched = state.get("fetched_pages") or []
    partials = [p["partial"] for p in fetched]
    if not partials:
        empty = PersonProfile(full_name=full_name, confidence="low").model_dump(mode="json")
        empty, _ = await get_guardrails().redact_profile(empty)
        return {"profile": empty, "phase": "confirm"}

    prompt = MERGE_PROFILE_PROMPT.format(
        full_name=full_name,
        distinguishers=distinguishers or "(none)",
        partials=json.dumps(partials, ensure_ascii=False, indent=2)[:30_000],
    )
    model = build_structured_model(PersonProfile, temperature=0.0)
    try:
        profile = await model.ainvoke(
            [SystemMessage(content=SYSTEM_RESEARCHER), HumanMessage(content=prompt)]
        )
    except Exception as exc:
        log.warning("build_profile_failed", error=str(exc))
        profile = PersonProfile(full_name=full_name, confidence="low")
    if not isinstance(profile, PersonProfile):
        # The merge model can return None (no tool call) or an odd shape; coerce
        # a dict, otherwise fall back rather than raising.
        try:
            profile = PersonProfile.model_validate(profile)
        except Exception as exc:
            log.warning("build_profile_invalid_result", error=str(exc))
            profile = PersonProfile(full_name=full_name, confidence="low")
    # Provenance is deterministically known from the partials; the merge LLM
    # drops the evidence array non-deterministically, so backfill it here.
    profile = _merge_evidence(profile, partials)
    # For a blocked page (empty body) the search snippet is the known, sole
    # source — carry it into evidence deterministically rather than relying on
    # the merge LLM to keep it.
    profile = _backfill_blocked_snippets(profile, fetched)
    profile_dict = profile.model_dump(mode="json")
    profile_dict, _ = await get_guardrails().redact_profile(profile_dict)
    return {"profile": profile_dict, "phase": "confirm"}


def _merge_evidence(
    profile: PersonProfile, partials: list[dict[str, Any]]
) -> PersonProfile:
    """Union every source URL the partials cited into the profile's evidence.

    The merge LLM sometimes returns an empty `evidence` list even though each
    partial carries its source. Since that provenance is already known, add any
    missing entries (deduped by URL) so the final profile always cites sources.
    """
    seen = {str(ev.url) for ev in profile.evidence}
    for partial in partials:
        for ev in partial.get("evidence") or []:
            url = ev.get("url")
            if not url or str(url) in seen:
                continue
            try:
                profile.evidence.append(Evidence.model_validate(ev))
            except Exception:
                continue
            seen.add(str(url))
    return profile


def _backfill_blocked_snippets(
    profile: PersonProfile, fetched_pages: list[dict[str, Any]]
) -> PersonProfile:
    """Deterministically attach a blocked page's search snippet to its evidence.

    When a page body could not be fetched (`fetch_blocked`), the search snippet
    is the known, sole source for that URL. Ensure the final profile's evidence
    entry for the URL carries that snippet text: fill it in if the entry exists
    without a snippet, or create the entry if the merge LLM omitted it entirely.
    A page whose body loaded is left untouched — its facts come from the body,
    not the search result.
    """
    by_url: dict[str, dict[str, Any]] = {}
    for page in fetched_pages:
        if not page.get("fetch_blocked"):
            continue
        url, snippet = page.get("url"), page.get("snippet")
        if url and snippet:
            by_url.setdefault(
                str(url).rstrip("/"),
                {"url": url, "platform": page.get("platform"), "snippet": snippet},
            )
    if not by_url:
        return profile

    existing = {str(ev.url).rstrip("/"): ev for ev in profile.evidence}
    for key, info in by_url.items():
        ev = existing.get(key)
        if ev is not None:
            if not ev.snippet:
                ev.snippet = info["snippet"]
            continue
        try:
            profile.evidence.append(
                Evidence(url=info["url"], platform=info["platform"], snippet=info["snippet"])
            )
        except Exception:
            continue
    return profile


async def confirm_profile(state: PeopleSearchState) -> dict[str, Any]:
    """Show the assembled profile and wait for the user's decision."""
    profile = state.get("profile")
    decision = interrupt(
        {
            "kind": "confirm_profile",
            "profile": profile,
            "candidates": list(state.get("candidates") or []),
            "locale": state.get("locale", "en"),
        }
    )
    next_phase: str = "done"
    user_decision = "approve"
    selected_index: int | None = None
    if isinstance(decision, dict):
        action = decision.get("decision") or decision.get("action") or "approve"
        if action == "more":
            user_decision = "more"
            next_phase = "narrow"
        elif action == "switch_candidate":
            user_decision = "switch_candidate"
            next_phase = "fetch"
            idx = decision.get("index")
            if isinstance(idx, int):
                selected_index = idx
        elif action == "abort":
            user_decision = "abort"
            next_phase = "done"
        else:
            user_decision = "approve"
            next_phase = "done"
    return {
        "user_decision": user_decision,
        "phase": next_phase,
        "selected_candidate_index": selected_index,
    }


def route_after_confirm(state: PeopleSearchState) -> str:
    phase = state.get("phase", "done")
    if phase == "narrow":
        return "narrow_query"
    if phase == "fetch":
        return "fetch_pages"
    return "__end__"


# ----- Utilities -----


def _safe_json(text: object) -> dict[str, Any] | None:
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return None
    # Strip code fences if the model wrapped JSON in them.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError:
        return None
