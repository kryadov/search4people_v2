"""Deterministic in-memory backend for tests (no torch)."""

from __future__ import annotations

from app.guardrails.types import GuardModel, Span


class FakeBackend:
    """Scripted classify/extract.

    `scores`: {label: score} returned (filtered to requested labels) by classify.
    `entities`: list of Span returned (filtered to requested types) by extract.
    """

    def __init__(
        self,
        scores: dict[str, float] | None = None,
        entities: list[Span] | None = None,
    ) -> None:
        self.scores = scores or {}
        self.entities = entities or []
        self.classify_calls: list[tuple[str, tuple[str, ...], GuardModel]] = []
        self.extract_calls: list[tuple[str, tuple[str, ...], GuardModel]] = []

    async def classify(
        self, text: str, labels: list[str], *, model: GuardModel = "safety"
    ) -> dict[str, float]:
        self.classify_calls.append((text, tuple(labels), model))
        return {label: self.scores.get(label, 0.0) for label in labels}

    async def extract(
        self, text: str, entity_types: list[str], *, model: GuardModel = "pii"
    ) -> list[Span]:
        self.extract_calls.append((text, tuple(entity_types), model))
        return [e for e in self.entities if e.label in entity_types]
