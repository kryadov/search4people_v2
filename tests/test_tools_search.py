"""Search adapter tests with DDGS stubbed."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.tools.search import search_many_platforms, search_platform, web_search


class _StubDDGS:
    def __init__(self, hits: list[dict]):
        self._hits = hits

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, query: str, **kwargs):
        return self._hits


@pytest.mark.asyncio
async def test_search_platform_uses_site_restriction():
    hits = [
        {"href": "https://github.com/torvalds", "title": "Linus Torvalds", "body": "Linux"},
        {"href": "https://github.com/torvalds", "title": "dup", "body": "dup"},
    ]
    with patch("app.tools.search.DDGS", lambda: _StubDDGS(hits)):
        results = await search_platform("github", "Linus Torvalds", limit=10)
    assert len(results) == 1
    assert results[0].url == "https://github.com/torvalds"
    assert results[0].platform == "github"


@pytest.mark.asyncio
async def test_web_search_returns_normalized_hits():
    hits = [
        {"href": "https://example.com/a", "title": "A", "body": "alpha"},
        {"href": "https://example.com/b", "title": "B", "body": "beta"},
    ]
    with patch("app.tools.search.DDGS", lambda: _StubDDGS(hits)):
        results = await web_search("x", limit=10)
    assert {r.url for r in results} == {"https://example.com/a", "https://example.com/b"}


@pytest.mark.asyncio
async def test_search_many_platforms_dedupes_across_platforms():
    hits = [{"href": "https://x.com/foo", "title": "foo", "body": "snip"}]
    with patch("app.tools.search.DDGS", lambda: _StubDDGS(hits)):
        results = await search_many_platforms(["twitter", "github"], "foo", limit_per_platform=5)
    # x.com matches the 'twitter' platform pattern; second platform yields nothing new.
    urls = [r.url for r in results]
    assert urls.count("https://x.com/foo") == 1
