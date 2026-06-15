"""Sidecar backend: a remote service exposing classify/extract over HTTP.

The wire contract mirrors the GuardBackend protocol:
  POST {base}/classify  {"text","labels","model"} -> {"scores": {label: float}}
  POST {base}/extract   {"text","entity_types","model"} -> {"spans": [Span,...]}
"""

from __future__ import annotations

import httpx

from app.guardrails.types import GuardModel, Span


class HttpBackend:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        if not base_url:
            raise RuntimeError("GUARDRAILS_HTTP_URL is required for backend=http")
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    async def classify(
        self, text: str, labels: list[str], *, model: GuardModel = "safety"
    ) -> dict[str, float]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/classify",
                json={"text": text, "labels": labels, "model": model},
            )
            resp.raise_for_status()
            data = resp.json()
        scores = data.get("scores") or {}
        return {label: float(scores.get(label, 0.0)) for label in labels}

    async def extract(
        self, text: str, entity_types: list[str], *, model: GuardModel = "pii"
    ) -> list[Span]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base}/extract",
                json={"text": text, "entity_types": entity_types, "model": model},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            Span(
                label=s["label"],
                start=int(s["start"]),
                end=int(s["end"]),
                text=str(s.get("text", "")),
                score=float(s.get("score", 1.0)),
            )
            for s in (data.get("spans") or [])
        ]
