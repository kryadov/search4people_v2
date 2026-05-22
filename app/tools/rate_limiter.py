"""Per-host async token bucket — keeps us polite to every domain we hit."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from urllib.parse import urlparse


class _HostBucket:
    """Simple token bucket with a refill rate of `rps` tokens per second."""

    __slots__ = ("capacity", "last_refill", "lock", "rps", "tokens")

    def __init__(self, rps: float, capacity: float | None = None) -> None:
        self.rps = max(0.01, rps)
        self.capacity = capacity if capacity is not None else max(1.0, rps)
        self.tokens = self.capacity
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rps)
                self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                deficit = 1.0 - self.tokens
                await asyncio.sleep(deficit / self.rps)


class RateLimiter:
    """One token bucket per hostname."""

    def __init__(self, rps: float) -> None:
        self._rps = rps
        self._buckets: dict[str, _HostBucket] = defaultdict(lambda: _HostBucket(self._rps))

    async def acquire(self, url: str) -> None:
        host = urlparse(url).hostname or ""
        await self._buckets[host].acquire()
