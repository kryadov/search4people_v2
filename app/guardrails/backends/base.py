"""Backend protocol: the raw classify/extract engine behind all detectors."""

from __future__ import annotations

from typing import Protocol

from app.guardrails.types import GuardModel, Span


class GuardBackend(Protocol):
    async def classify(
        self, text: str, labels: list[str], *, model: GuardModel = "safety"
    ) -> dict[str, float]:
        """Return a score in [0,1] for each requested label."""
        ...

    async def extract(
        self, text: str, entity_types: list[str], *, model: GuardModel = "pii"
    ) -> list[Span]:
        """Return NER spans for the requested entity types."""
        ...
