"""Pure mapping from detector findings to an enforcement verdict."""

from __future__ import annotations

from app.guardrails.types import (
    ACTION_PRIORITY,
    GuardAction,
    GuardCategory,
    GuardFinding,
    GuardVerdict,
)

Policy = dict[GuardCategory, tuple[GuardAction, float]]


def apply(findings: list[GuardFinding], policy: Policy, text: str) -> GuardVerdict:
    """Drop sub-threshold findings, pick the worst action, build transformed text."""
    kept: list[GuardFinding] = []
    action: GuardAction = "allow"
    for f in findings:
        rule = policy.get(f.category)
        if rule is None:
            continue
        act, threshold = rule
        if f.score < threshold:
            continue
        kept.append(f)
        if ACTION_PRIORITY[act] > ACTION_PRIORITY[action]:
            action = act

    transformed: str | None = None
    if action in ("sanitize", "redact"):
        transformed = _rewrite(text, kept, policy)

    reason = ", ".join(sorted({f.category for f in kept})) if kept else ""
    return GuardVerdict(
        action=action, findings=kept, transformed_text=transformed, reason=reason
    )


def _rewrite(text: str, findings: list[GuardFinding], policy: Policy) -> str:
    """Replace finding spans. PII -> [redacted:<label>]; injection -> [removed]."""
    edits: list[tuple[int, int, str]] = []
    for f in findings:
        act = policy[f.category][0]
        if act == "redact":
            token = f"[redacted:{f.label}]"
        elif act == "sanitize":
            token = "[removed]"
        else:
            continue
        for start, end in f.spans:
            edits.append((start, end, token))
    # Apply right-to-left so earlier offsets stay valid.
    out = text
    for start, end, token in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + token + out[end:]
    return out
