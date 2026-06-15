"""Detectors translate raw backend scores/spans into typed GuardFindings."""

from __future__ import annotations

from app.guardrails.backends.base import GuardBackend
from app.guardrails.types import GuardCategory, GuardFinding, Span

# --- Safety (gliner-guard-omni) label families -> our categories ---
_INJECTION_LABELS: dict[str, GuardCategory] = {
    "prompt_injection": "prompt_injection",
    "instruction_override": "prompt_injection",
    "data_exfiltration": "prompt_injection",
}
_JAILBREAK_LABELS: dict[str, GuardCategory] = {
    "jailbreak_persona": "jailbreak",
    "jailbreak": "jailbreak",
}
_HARMFUL_LABELS: dict[str, GuardCategory] = {
    "harassment": "harmful_intent",
    "hate_speech": "harmful_intent",
    "violence": "harmful_intent",
    "child_exploitation": "minor_target",
    "stalking": "harmful_intent",
    "doxxing": "harmful_intent",
}
_TOXICITY_LABELS: dict[str, GuardCategory] = {"toxicity": "toxicity"}

# Entity types the PII detector asks the NER model for (sensitive contact/financial
# data only -- NOT the target person's name, which is the whole point of the search).
PII_ENTITY_TYPES = [
    "email",
    "phone",
    "address",
    "id_number",
    "passport",
    "card_number",
    "bank_account",
    "crypto_wallet",
    "social_account",
]

# OSINT intent labels for the general gliner2 model (zero-shot classification).
_INTENT_LABELS: dict[str, GuardCategory] = {
    "stalking": "harmful_intent",
    "doxxing": "harmful_intent",
    "harassment": "harmful_intent",
    "minor": "minor_target",
}


class SafetyDetector:
    """gliner-guard-omni: jailbreak / injection / harmful / toxicity."""

    def __init__(self, backend: GuardBackend) -> None:
        self._backend = backend

    async def detect(self, text: str, *, injection_only: bool = False) -> list[GuardFinding]:
        if injection_only:
            label_map: dict[str, GuardCategory] = dict(_INJECTION_LABELS)
        else:
            label_map = {
                **_INJECTION_LABELS,
                **_JAILBREAK_LABELS,
                **_HARMFUL_LABELS,
                **_TOXICITY_LABELS,
            }
        scores = await self._backend.classify(text, list(label_map), model="safety")
        out: list[GuardFinding] = []
        for label, score in scores.items():
            category = label_map.get(label)
            if category is None or score <= 0.0:
                continue
            out.append(GuardFinding(category=category, score=score, label=label))
        return out


class PIIDetector:
    """gliner2 NER over sensitive contact/financial entity types."""

    def __init__(self, backend: GuardBackend) -> None:
        self._backend = backend

    async def detect(self, text: str) -> list[GuardFinding]:
        spans: list[Span] = await self._backend.extract(
            text, PII_ENTITY_TYPES, model="pii"
        )
        return [
            GuardFinding(
                category="pii", score=s.score, label=s.label, spans=[(s.start, s.end)]
            )
            for s in spans
        ]


class IntentDetector:
    """gliner2 zero-shot classification of OSINT request intent."""

    def __init__(self, backend: GuardBackend) -> None:
        self._backend = backend

    async def detect(self, text: str) -> list[GuardFinding]:
        scores = await self._backend.classify(text, list(_INTENT_LABELS), model="pii")
        out: list[GuardFinding] = []
        for label, score in scores.items():
            category = _INTENT_LABELS.get(label)
            if category is None or score <= 0.0:
                continue
            out.append(GuardFinding(category=category, score=score, label=label))
        return out
