"""build_profile must deterministically carry a blocked page's snippet into evidence.

When a candidate's page body is blocked (robots/auth-wall), the search snippet is
the known, sole source. The merge LLM may emit evidence without the snippet text
(or omit it entirely); build_profile backfills it in code so blocked-page
provenance is deterministic, not at the LLM's discretion.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph import nodes
from app.models.profile import Evidence, PersonProfile

_URL = "https://www.linkedin.com/in/konstantin-riadov"
_SNIPPET = "Senior Engineer at Acme. Based in Berlin."


def _blocked_page(partial: PersonProfile) -> dict:
    return {
        "url": _URL,
        "platform": "linkedin",
        "snippet": _SNIPPET,
        "markdown_len": 0,
        "fetch_blocked": True,
        "partial": partial.model_dump(mode="json"),
    }


@pytest.mark.asyncio
async def test_backfills_snippet_when_evidence_url_present_without_snippet(monkeypatch):
    # Merge LLM kept the URL but dropped the snippet text.
    merged = PersonProfile(
        full_name="Konstantin Riadov",
        evidence=[Evidence(url=_URL, platform="linkedin")],
        confidence="low",
    )
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=merged)
    monkeypatch.setattr(nodes, "build_structured_model", lambda *a, **k: fake_model)

    partial = PersonProfile(full_name="Konstantin Riadov", confidence="low")
    state = {
        "query": {"first_name": "Konstantin", "last_name": "Riadov"},
        "fetched_pages": [_blocked_page(partial)],
    }

    patch = await nodes.build_profile(state)
    profile = PersonProfile.model_validate(patch["profile"])

    ev = [e for e in profile.evidence if "konstantin-riadov" in str(e.url)]
    assert len(ev) == 1
    assert ev[0].snippet == _SNIPPET


@pytest.mark.asyncio
async def test_creates_evidence_when_llm_omits_it_for_blocked_page(monkeypatch):
    # Neither the merge LLM nor the partial carry any evidence for the URL.
    merged = PersonProfile(full_name="Konstantin Riadov", confidence="low")
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=merged)
    monkeypatch.setattr(nodes, "build_structured_model", lambda *a, **k: fake_model)

    partial = PersonProfile(full_name="Konstantin Riadov", confidence="low")
    state = {
        "query": {"first_name": "Konstantin", "last_name": "Riadov"},
        "fetched_pages": [_blocked_page(partial)],
    }

    patch = await nodes.build_profile(state)
    profile = PersonProfile.model_validate(patch["profile"])

    ev = [e for e in profile.evidence if "konstantin-riadov" in str(e.url)]
    assert len(ev) == 1
    assert ev[0].snippet == _SNIPPET
    assert ev[0].platform == "linkedin"


@pytest.mark.asyncio
async def test_does_not_backfill_search_snippet_for_a_fetched_page(monkeypatch):
    # A page whose body DID load must not have the search snippet forced into
    # evidence — its facts come from the body, not the search result.
    merged = PersonProfile(
        full_name="Jane Doe",
        evidence=[Evidence(url="https://github.com/jane-doe", platform="github")],
        confidence="medium",
    )
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=merged)
    monkeypatch.setattr(nodes, "build_structured_model", lambda *a, **k: fake_model)

    partial = PersonProfile(
        full_name="Jane Doe",
        evidence=[Evidence(url="https://github.com/jane-doe", platform="github")],
        confidence="medium",
    )
    state = {
        "query": {"first_name": "Jane", "last_name": "Doe"},
        "fetched_pages": [
            {
                "url": "https://github.com/jane-doe",
                "platform": "github",
                "snippet": "an irrelevant search snippet",
                "markdown_len": 500,
                "fetch_blocked": False,
                "partial": partial.model_dump(mode="json"),
            }
        ],
    }

    patch = await nodes.build_profile(state)
    profile = PersonProfile.model_validate(patch["profile"])

    ev = [e for e in profile.evidence if "jane-doe" in str(e.url)]
    assert len(ev) == 1
    assert ev[0].snippet is None


@pytest.mark.asyncio
async def test_build_profile_survives_merge_model_returning_none(monkeypatch):
    # The merge model can return None (no tool call); build_profile must not raise.
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=None)
    monkeypatch.setattr(nodes, "build_structured_model", lambda *a, **k: fake_model)

    partial = PersonProfile(full_name="Konstantin Riadov", confidence="low")
    state = {
        "query": {"first_name": "Konstantin", "last_name": "Riadov"},
        "fetched_pages": [_blocked_page(partial)],
    }

    patch = await nodes.build_profile(state)
    profile = PersonProfile.model_validate(patch["profile"])

    assert profile.full_name == "Konstantin Riadov"
    # The blocked page's snippet is still carried into evidence deterministically.
    ev = [e for e in profile.evidence if "konstantin-riadov" in str(e.url)]
    assert len(ev) == 1
    assert ev[0].snippet == _SNIPPET
