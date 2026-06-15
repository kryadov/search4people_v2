"""Guardrails value types shared across backends, detectors, policy, engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

GuardCategory = Literal[
    "jailbreak",
    "prompt_injection",
    "harmful_intent",
    "minor_target",
    "toxicity",
    "pii",
]
GuardAction = Literal["allow", "block", "sanitize", "redact", "flag"]
GuardModel = Literal["safety", "pii"]

# Action precedence for aggregating multiple findings into one verdict.
ACTION_PRIORITY: dict[GuardAction, int] = {
    "allow": 0,
    "flag": 1,
    "redact": 2,
    "sanitize": 3,
    "block": 4,
}


@dataclass(frozen=True)
class Span:
    """A character range emitted by NER extraction."""

    label: str
    start: int
    end: int
    text: str
    score: float = 1.0


@dataclass
class GuardFinding:
    category: GuardCategory
    score: float
    label: str
    spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class GuardVerdict:
    action: GuardAction
    findings: list[GuardFinding] = field(default_factory=list)
    transformed_text: str | None = None
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"
