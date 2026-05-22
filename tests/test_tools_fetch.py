"""Fetch tests with httpx mocked via respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from app.tools.fetch import PageFetcher


@pytest.mark.asyncio
@respx.mock
async def test_fetch_returns_markdown_from_html(monkeypatch):
    monkeypatch.setenv("JS_HEAVY_DOMAINS", "")  # avoid Playwright in tests
    from app.config import get_settings

    get_settings.cache_clear()

    respx.get("https://robots.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
    )
    respx.get("https://robots.example.com/page").mock(
        return_value=httpx.Response(
            200,
            text=(
                "<html><body><h1>Jane Doe</h1>"
                + "<p>"
                + ("Bio paragraph. " * 200)
                + "</p>"
                + "</body></html>"
            ),
        )
    )

    fetcher = PageFetcher()
    try:
        result = await fetcher.fetch("https://robots.example.com/page")
    finally:
        await fetcher.close()

    assert result.status == 200
    assert result.used_playwright is False
    assert "Jane Doe" in result.markdown


@pytest.mark.asyncio
@respx.mock
async def test_fetch_respects_robots(monkeypatch):
    monkeypatch.setenv("JS_HEAVY_DOMAINS", "")
    from app.config import get_settings

    get_settings.cache_clear()

    respx.get("https://blocked.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text="User-agent: *\nDisallow: /\n")
    )

    fetcher = PageFetcher()
    try:
        result = await fetcher.fetch("https://blocked.example.com/secret")
    finally:
        await fetcher.close()

    assert result.blocked_by_robots is True
    assert result.markdown == ""
