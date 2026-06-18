"""fetch_pages must not discard a blocked candidate that carries a snippet."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph import nodes
from app.models.profile import Evidence, PersonProfile
from app.tools.fetch import FetchResult


def _noop_guardrails():
    g = MagicMock()
    # scan_content returns an object whose transformed_text is None (no change).
    g.scan_content = AsyncMock(return_value=MagicMock(transformed_text=None))
    return g


@pytest.mark.asyncio
async def test_blocked_candidate_with_snippet_is_kept(monkeypatch):
    # Fetcher returns an empty body (robots/auth-wall blocked).
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock(return_value=FetchResult(
        url="https://www.linkedin.com/in/konstantin-riadov",
        status=0,
        markdown="",
        used_playwright=False,
        blocked_by_robots=True,
    ))
    monkeypatch.setattr(nodes, "_get_fetcher", lambda: fetcher)
    monkeypatch.setattr(nodes, "get_guardrails", _noop_guardrails)

    captured: dict = {}

    async def fake_extract(**kwargs):
        captured.update(kwargs)
        return PersonProfile(
            full_name="Konstantin Riadov",
            bio="Senior Engineer at Acme, Berlin.",
            evidence=[Evidence(
                url="https://www.linkedin.com/in/konstantin-riadov",
                platform="linkedin",
                snippet="Senior Engineer at Acme. Based in Berlin.",
            )],
            confidence="low",
        )

    monkeypatch.setattr(nodes, "extract_profile_from_page", fake_extract)

    state = {
        "query": {"first_name": "Konstantin", "last_name": "Riadov"},
        "candidates": [{
            "url": "https://www.linkedin.com/in/konstantin-riadov",
            "title": "Konstantin Riadov - Senior Engineer at Acme - Berlin",
            "snippet": "Senior Engineer at Acme. Based in Berlin.",
            "platform": "linkedin",
        }],
        "selected_candidate_index": 0,
        "visited_urls": [],
        "fetched_pages": [],
    }

    patch = await nodes.fetch_pages(state)

    pages = patch["fetched_pages"]
    assert len(pages) == 1
    assert pages[0]["fetch_blocked"] is True
    assert pages[0]["markdown_len"] == 0
    # The snippet+title reached extraction.
    assert captured["snippet"] == "Senior Engineer at Acme. Based in Berlin."
    assert captured["title"] == "Konstantin Riadov - Senior Engineer at Acme - Berlin"


@pytest.mark.asyncio
async def test_blocked_candidate_without_snippet_is_dropped(monkeypatch):
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock(return_value=FetchResult(
        url="https://example.com/nothing",
        status=0,
        markdown="",
        used_playwright=False,
    ))
    monkeypatch.setattr(nodes, "_get_fetcher", lambda: fetcher)
    monkeypatch.setattr(nodes, "get_guardrails", _noop_guardrails)

    async def fake_extract(**kwargs):  # must not be called
        raise AssertionError("extract should not run when there is nothing to extract")

    monkeypatch.setattr(nodes, "extract_profile_from_page", fake_extract)

    state = {
        "query": {"first_name": "Jane", "last_name": "Doe"},
        "candidates": [{
            "url": "https://example.com/nothing",
            "title": None,
            "snippet": None,
            "platform": None,
        }],
        "selected_candidate_index": 0,
        "visited_urls": [],
        "fetched_pages": [],
    }

    patch = await nodes.fetch_pages(state)
    assert patch["fetched_pages"] == []
