"""No-op backend: always-empty results, zero dependencies."""

from __future__ import annotations

from app.guardrails.types import GuardModel, Span


class NoOpBackend:
    async def classify(
        self, text: str, labels: list[str], *, model: GuardModel = "safety"
    ) -> dict[str, float]:
        return {label: 0.0 for label in labels}

    async def extract(
        self, text: str, entity_types: list[str], *, model: GuardModel = "pii"
    ) -> list[Span]:
        return []
