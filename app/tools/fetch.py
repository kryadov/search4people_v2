"""Page fetcher.

Strategy:
1. Respect robots.txt (configurable user-agent).
2. Apply per-host rate limit.
3. Try `httpx` first. If the response is 403/blocked/empty or the host is in
   `JS_HEAVY_DOMAINS`, fall back to Playwright (headless Chromium).
4. Convert HTML → Markdown via `markdownify` for LLM-friendly downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import structlog
from markdownify import markdownify as md
from selectolax.parser import HTMLParser

from app.config import get_settings
from app.tools.rate_limiter import RateLimiter
from app.tools.robots import RobotsChecker

log = structlog.get_logger()

_JS_FAILURE_STATUSES = {401, 403, 451, 503}
_MIN_BODY_CHARS = 400


@dataclass(slots=True)
class FetchResult:
    url: str
    status: int
    markdown: str
    used_playwright: bool
    blocked_by_robots: bool = False
    error: str | None = None


class PageFetcher:
    def __init__(self) -> None:
        settings = get_settings()
        self._user_agent = settings.user_agent
        self._js_heavy = {d.lower() for d in settings.js_heavy_domains}
        self._rate_limiter = RateLimiter(settings.per_host_rps)
        self._robots = RobotsChecker(self._user_agent)
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                http2=True,
                follow_redirects=True,
                timeout=httpx.Timeout(15.0, connect=8.0),
                headers={"User-Agent": self._user_agent},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _is_js_heavy(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in self._js_heavy)

    async def fetch(self, url: str) -> FetchResult:
        if not await self._robots.allowed(url):
            log.info("robots_blocked", url=url)
            return FetchResult(
                url=url,
                status=0,
                markdown="",
                used_playwright=False,
                blocked_by_robots=True,
            )

        await self._rate_limiter.acquire(url)

        if self._is_js_heavy(url):
            return await self._fetch_playwright(url)

        client = await self._ensure_client()
        try:
            resp = await client.get(url)
        except httpx.HTTPError as exc:
            log.warning("httpx_failed", url=url, error=str(exc))
            return await self._fetch_playwright(url, prior_error=str(exc))

        body = resp.text or ""
        markdown = _html_to_markdown(body)
        if resp.status_code in _JS_FAILURE_STATUSES or len(markdown) < _MIN_BODY_CHARS:
            log.info(
                "fallback_to_playwright",
                url=url,
                status=resp.status_code,
                length=len(markdown),
            )
            return await self._fetch_playwright(url)

        return FetchResult(
            url=url,
            status=resp.status_code,
            markdown=markdown,
            used_playwright=False,
        )

    async def _fetch_playwright(self, url: str, prior_error: str | None = None) -> FetchResult:
        # Imported lazily so unit tests don't require browsers.
        from playwright.async_api import async_playwright

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(user_agent=self._user_agent)
                    page = await context.new_page()
                    response = await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    status = response.status if response else 0
                    html = await page.content()
                finally:
                    await browser.close()
        except Exception as exc:
            log.warning("playwright_failed", url=url, error=str(exc))
            return FetchResult(
                url=url,
                status=0,
                markdown="",
                used_playwright=True,
                error=prior_error or str(exc),
            )

        return FetchResult(
            url=url,
            status=status,
            markdown=_html_to_markdown(html),
            used_playwright=True,
        )


def _html_to_markdown(html: str) -> str:
    if not html:
        return ""
    # Strip script/style/nav noise before markdownify to keep output tight.
    try:
        tree = HTMLParser(html)
        for selector in ("script", "style", "nav", "footer", "header", "noscript"):
            for node in tree.css(selector):
                node.decompose()
        cleaned = tree.body.html if tree.body else html
    except Exception:
        cleaned = html
    return md(cleaned or html, heading_style="ATX", strip=["img"]).strip()
