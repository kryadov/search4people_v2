"""LLM-driven structured extraction from fetched page markdown."""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import build_chat_model
from app.models.profile import PersonProfile

log = structlog.get_logger()

_SYSTEM = """You extract structured biographical facts about a single specific person from one web page.

You will receive:
- the person's first and last name (and possibly other distinguishing attributes), and
- a Markdown rendering of a web page that may or may not be about that person.

Return a partial PersonProfile containing ONLY fields that the page actually
supports. Leave fields you cannot back out of the text empty or null. Always
populate `evidence` with at least one entry citing the page URL. Do not
fabricate.

If the page is clearly NOT about this person, return an empty PersonProfile
with `full_name` set to the queried name and `confidence: "low"`.
"""


async def extract_profile_from_page(
    *,
    full_name: str,
    distinguishers: str,
    url: str,
    markdown: str,
    platform: str | None,
    max_chars: int = 12_000,
) -> PersonProfile:
    """Run the LLM in structured-output mode against a single fetched page."""
    body = markdown[:max_chars]
    model = build_chat_model(temperature=0.0).with_structured_output(PersonProfile)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"Person: {full_name}\n"
                f"Distinguishing attributes: {distinguishers or '(none)'}\n"
                f"Source URL: {url}\n"
                f"Source platform: {platform or 'unknown'}\n\n"
                f"--- PAGE MARKDOWN ---\n{body}"
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
    # Some providers return a dict; coerce defensively.
    return PersonProfile.model_validate(result)
