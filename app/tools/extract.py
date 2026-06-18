"""LLM-driven structured extraction from fetched page markdown."""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import build_structured_model
from app.models.profile import PersonProfile

log = structlog.get_logger()

_SYSTEM = """You extract structured biographical facts about a single specific person from one web page.

You will receive:
- the person's first and last name (and possibly other distinguishing attributes),
- the search-result title and snippet for the page (a trustworthy summary), and
- a Markdown rendering of a web page that may or may not be about that person.

The search-result title and snippet are a legitimate source: when the page body
is empty or is a login / placeholder / "join to view" wall, rely on the title
and snippet. Do not fabricate beyond what the snippet and page support.

Return a partial PersonProfile containing ONLY fields that the page or the
search result actually supports. Leave fields you cannot back out of the text
empty or null. Always populate `evidence` with at least one entry citing the
page URL; when a fact comes from the snippet, set that evidence entry's
`snippet`. Do not fabricate.

If the page and the search result are clearly NOT about this person, return an
empty PersonProfile with `full_name` set to the queried name and
`confidence: "low"`.
"""


async def extract_profile_from_page(
    *,
    full_name: str,
    distinguishers: str,
    url: str,
    markdown: str,
    platform: str | None,
    title: str | None = None,
    snippet: str | None = None,
    max_chars: int = 12_000,
) -> PersonProfile:
    """Run the LLM in structured-output mode against a single fetched page."""
    body = markdown[:max_chars]
    search_lines: list[str] = []
    if title:
        search_lines.append(f"Title: {title}")
    if snippet:
        search_lines.append(f"Snippet: {snippet}")
    search_block = "\n".join(search_lines) or "(none)"
    model = build_structured_model(PersonProfile, temperature=0.0)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"Person: {full_name}\n"
                f"Distinguishing attributes: {distinguishers or '(none)'}\n"
                f"Source URL: {url}\n"
                f"Source platform: {platform or 'unknown'}\n\n"
                f"--- SEARCH RESULT ---\n{search_block}\n\n"
                f"--- PAGE MARKDOWN ---\n{body or '(page body unavailable)'}"
            )
        ),
    ]
    try:
        result = await model.ainvoke(messages)
    except Exception as exc:
        log.warning("extract_failed", url=url, error=str(exc))
        return PersonProfile(full_name=full_name, confidence="low")
    if isinstance(result, PersonProfile):
        return result
    # Structured-output models can return None (no tool call) or an unexpected
    # shape — e.g. on a blocked page with an empty body. Coerce a dict; fall
    # back to a low-confidence profile for anything else rather than raising.
    if isinstance(result, dict):
        try:
            return PersonProfile.model_validate(result)
        except Exception as exc:
            log.warning("extract_invalid_result", url=url, error=str(exc))
    else:
        log.warning("extract_empty_result", url=url, result_type=type(result).__name__)
    return PersonProfile(full_name=full_name, confidence="low")
