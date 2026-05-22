"""Cached robots.txt checker. Soft-fails to `allowed` if the file can't be loaded
so a single broken host doesn't block the whole flow."""

from __future__ import annotations

from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import structlog

log = structlog.get_logger()


class RobotsChecker:
    def __init__(self, user_agent: str, timeout: float = 5.0) -> None:
        self._user_agent = user_agent
        self._timeout = timeout
        self._cache: dict[str, RobotFileParser] = {}

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.hostname:
            return False
        host_key = f"{parsed.scheme}://{parsed.hostname}"
        parser = self._cache.get(host_key)
        if parser is None:
            parser = await self._load(host_key)
            self._cache[host_key] = parser
        try:
            return parser.can_fetch(self._user_agent, url)
        except Exception:  # pragma: no cover - robotparser quirk
            return True

    async def _load(self, host_key: str) -> RobotFileParser:
        parser = RobotFileParser()
        robots_url = urljoin(host_key + "/", "robots.txt")
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(robots_url, headers={"User-Agent": self._user_agent})
            if response.status_code >= 400:
                parser.parse([])
                return parser
            parser.parse(response.text.splitlines())
        except (httpx.HTTPError, OSError) as exc:
            log.debug("robots_fetch_failed", url=robots_url, error=str(exc))
            parser.parse([])
        return parser
