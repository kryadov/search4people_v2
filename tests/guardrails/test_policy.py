from app.guardrails.policy import apply
from app.guardrails.types import GuardFinding

POLICY = {
    "harmful_intent": ("block", 0.6),
    "prompt_injection": ("sanitize", 0.6),
    "pii": ("redact", 0.5),
    "toxicity": ("flag", 0.7),
}


def test_below_threshold_dropped_returns_allow():
    v = apply([GuardFinding("harmful_intent", 0.5, "stalking")], POLICY, "hi")
    assert v.action == "allow"
    assert v.findings == []


def test_block_wins_over_others():
    findings = [
        GuardFinding("pii", 0.9, "email", [(0, 5)]),
        GuardFinding("harmful_intent", 0.8, "stalking"),
    ]
    v = apply(findings, POLICY, "hi")
    assert v.action == "block"


def test_sanitize_replaces_spans():
    v = apply(
        [GuardFinding("prompt_injection", 0.9, "instruction_override", [(0, 6)])],
        POLICY,
        "ignore the rest",
    )
    assert v.action == "sanitize"
    assert v.transformed_text == "[removed] the rest"


def test_redact_replaces_with_typed_token():
    text = "mail me x@y.com now"
    v = apply([GuardFinding("pii", 0.9, "email", [(8, 15)])], POLICY, text)
    assert v.action == "redact"
    assert v.transformed_text == "mail me [redacted:email] now"


def test_overlapping_spans_applied_right_to_left():
    text = "abcdefgh"
    findings = [
        GuardFinding("pii", 0.9, "a", [(0, 2)]),
        GuardFinding("pii", 0.9, "b", [(4, 6)]),
    ]
    v = apply(findings, POLICY, text)
    assert v.transformed_text == "[redacted:a]cd[redacted:b]gh"
