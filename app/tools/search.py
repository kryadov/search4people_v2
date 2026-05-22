"""Search abstractions.

Two public callables:
- `search_platform(platform, query, limit)`: site-restricted search for one of
  the configured platforms (e.g. `site:linkedin.com/in "John Doe"`).
- `web_search(query, limit)`: general open-web search.

Each tries the configured providers in priority order and merges the results,
de-duplicating by URL.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from urllib.parse import urlparse

import structlog
from ddgs import DDGS

from app.config import get_settings

log = structlog.get_logger()

# Per-platform `site:` patterns. Tuned to surface profile pages, not noise.
PLATFORM_SITES: dict[str, list[str]] = {
    "linkedin": ["linkedin.com/in", "linkedin.com/pub"],
    "github": ["github.com"],
    "twitter": ["twitter.com", "x.com"],
    "facebook": ["facebook.com"],
    "instagram": ["instagram.com"],
    "vk": ["vk.com"],
}


@dataclass(slots=True)
class SearchHit:
    url: str
    title: str | None
    snippet: str | None
    platform: str | None


def _build_site_query(query: str, platform: str) -> str:
    sites = PLATFORM_SITES.get(platform, [])
    if not sites:
        return query
    site_clause = " OR ".join(f"site:{s}" for s in sites)
    return f"({site_clause}) {query}"


def _dedupe(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    out: list[SearchHit] = []
    for hit in hits:
        if hit.url in seen:
            continue
        seen.add(hit.url)
        out.append(hit)
    return out


# ----- Provider adapters -----


async def _tavily_search(
    query: str, limit: int, include_domains: list[str] | None
) -> list[SearchHit]:
    settings = get_settings()
    if not settings.tavily_api_key:
        return []
    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return []
    tool = TavilySearch(
        tavily_api_key=settings.tavily_api_key,
        max_results=limit,
        search_depth="basic",
    )
    payload: dict[str, object] = {"query": query}
    if include_domains:
        payload["include_domains"] = include_domains
    try:
        raw = await asyncio.to_thread(tool.invoke, payload)
    except Exception as exc:
        log.warning("tavily_failed", error=str(exc))
        return []
    items = raw.get("results", []) if isinstance(raw, dict) else []
    return [
        SearchHit(
            url=item.get("url", ""),
            title=item.get("title"),
            snippet=item.get("content"),
            platform=_infer_platform(item.get("url", "")),
        )
        for item in items
        if item.get("url")
    ]


async def _ddg_search(query: str, limit: int) -> list[SearchHit]:
    def _run() -> list[dict]:
        try:
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=limit, region="wt-wt") or []
        except Exception as exc:
            log.warning("ddg_failed", error=str(exc))
            return []

    raw = await asyncio.to_thread(_run)
    return [
        SearchHit(
            url=item.get("href") or item.get("url", ""),
            title=item.get("title"),
            snippet=item.get("body"),
            platform=_infer_platform(item.get("href") or item.get("url", "")),
        )
        for item in raw
        if item.get("href") or item.get("url")
    ]


def _infer_platform(url: str) -> str | None:
    if not url:
        return None
    host = (urlparse(url).hostname or "").lower()
    for platform, patterns in PLATFORM_SITES.items():
        for pat in patterns:
            domain = pat.split("/")[0]
            if host == domain or host.endswith("." + domain):
                return platform
    return None


# ----- Public API -----


async def search_platform(platform: str, query: str, limit: int = 10) -> list[SearchHit]:
    """Search the open web restricted to `platform`'s domains."""
    settings = get_settings()
    site_query = _build_site_query(query, platform)
    sites = PLATFORM_SITES.get(platform, [])
    include_domains = [s.split("/")[0] for s in sites] if sites else None

    hits: list[SearchHit] = []
    for provider in settings.search_providers:
        if provider == "tavily":
            hits.extend(await _tavily_search(site_query, limit, include_domains))
        elif provider == "ddg":
            hits.extend(await _ddg_search(site_query, limit))
        if len(_dedupe(hits)) >= limit:
            break
    # Re-stamp platform since we know which one we asked for.
    for hit in hits:
        if hit.platform is None:
            hit.platform = platform
    return _dedupe(hits)[:limit]


async def web_search(query: str, limit: int = 10) -> list[SearchHit]:
    """General open-web search across all configured providers."""
    settings = get_settings()
    hits: list[SearchHit] = []
    for provider in settings.search_providers:
        if provider == "tavily":
            hits.extend(await _tavily_search(query, limit, None))
        elif provider == "ddg":
            hits.extend(await _ddg_search(query, limit))
        if len(_dedupe(hits)) >= limit:
            break
    return _dedupe(hits)[:limit]


async def search_many_platforms(
    platforms: list[str], query: str, limit_per_platform: int = 5
) -> list[SearchHit]:
    """Run `search_platform` for every platform in parallel and merge results."""
    results = await asyncio.gather(
        *(search_platform(p, query, limit_per_platform) for p in platforms),
        return_exceptions=True,
    )
    merged: list[SearchHit] = []
    for r in results:
        if isinstance(r, BaseException):
            log.warning("platform_search_failed", error=str(r))
            continue
        merged.extend(r)
    return _dedupe(merged)
