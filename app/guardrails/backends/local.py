"""In-process gliner2 backend. Heavy deps (transformers/torch) load lazily."""

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Any

from app.guardrails.config import GuardrailsSettings
from app.guardrails.types import GuardModel, Span

# Model-level inclusion floor. Kept below every policy threshold (all >= 0.5) so
# the model always returns anything the policy might act on; the per-category
# policy thresholds remain the single source of truth.
_MODEL_THRESHOLD = 0.3

# Single task name passed to classify_text; the score dict is keyed under it.
_TASK = "guard"


class LocalGlinerBackend:
    def __init__(self, settings: GuardrailsSettings) -> None:
        self._settings = settings
        self._models: dict[GuardModel, Any] = {}

    def _model(self, which: GuardModel) -> Any:
        if which not in self._models:
            from gliner2 import GLiNER2  # lazy: keeps base import torch-free

            name = (
                self._settings.safety_model
                if which == "safety"
                else self._settings.pii_model
            )
            kwargs: dict[str, Any] = {}
            if self._settings.device != "auto":
                kwargs["map_location"] = self._settings.device
            # gliner2 prints a config banner with an emoji on load, which raises
            # UnicodeEncodeError on a non-UTF-8 console (e.g. Windows cp1252).
            # Swallow that banner so model loading never crashes on stdout.
            with contextlib.redirect_stdout(io.StringIO()):
                self._models[which] = GLiNER2.from_pretrained(name, **kwargs)
        return self._models[which]

    async def classify(
        self, text: str, labels: list[str], *, model: GuardModel = "safety"
    ) -> dict[str, float]:
        return await asyncio.to_thread(self._classify_sync, text, labels, model)

    def _classify_sync(
        self, text: str, labels: list[str], model: GuardModel
    ) -> dict[str, float]:
        m = self._model(model)
        result = m.classify_text(
            text,
            {_TASK: {"labels": labels, "multi_label": True}},
            threshold=_MODEL_THRESHOLD,
            include_confidence=True,
        )
        return _parse_classify(result.get(_TASK), labels)

    async def extract(
        self, text: str, entity_types: list[str], *, model: GuardModel = "pii"
    ) -> list[Span]:
        return await asyncio.to_thread(self._extract_sync, text, entity_types, model)

    def _extract_sync(
        self, text: str, entity_types: list[str], model: GuardModel
    ) -> list[Span]:
        m = self._model(model)
        result = m.extract_entities(
            text,
            entity_types,
            threshold=_MODEL_THRESHOLD,
            include_spans=True,
            include_confidence=True,
        )
        out: list[Span] = []
        for label, items in (result.get("entities") or {}).items():
            for it in items:
                if isinstance(it, dict) and "start" in it and "end" in it:
                    out.append(
                        Span(
                            label=label,
                            start=int(it["start"]),
                            end=int(it["end"]),
                            text=str(it.get("text", "")),
                            score=float(it.get("confidence", 1.0)),
                        )
                    )
        return out


def _parse_classify(value: Any, labels: list[str]) -> dict[str, float]:
    """Normalize gliner2 multi-label output into {label: score}."""
    scores = {label: 0.0 for label in labels}
    if value is None:
        return scores
    items = value if isinstance(value, list) else [value]
    for it in items:
        if isinstance(it, dict) and "label" in it:
            scores[it["label"]] = float(it.get("confidence", 1.0))
        elif isinstance(it, str):
            scores[it] = 1.0
    return scores
