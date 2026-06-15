# Guardrails on GLiNER2 + hivetrace/gliner-guard-omni

**Date:** 2026-06-15
**Status:** Design approved, ready for implementation plan
**Branch:** feat/a2a-server (or a fresh `feat/guardrails`)

## Summary

Add a guardrails layer to the people-search agent that protects against abuse
requests, prompt injection, output PII leakage, and provides a compliance audit
trail. The layer composes two encoder-based models behind a pluggable backend:

- **hivetrace/gliner-guard-omni** — safety classification (jailbreak,
  prompt-injection, harmful intent, toxicity).
- **GLiNER2** — PII detection via NER and OSINT-specific intent classification
  (zero-shot).

Guardrails are **enabled by default** (`backend="local"`); the base install is
full and ships the model dependencies. A `noop` backend allows explicit opt-out
and underpins tests.

## Goals

1. **Abuse protection** — block requests with clear harmful intent (stalking,
   doxing, harassment, searching for minors). Follows the README ethics
   disclaimer.
2. **Anti-prompt-injection** — protect LLM prompts from injection in untrusted
   content (fetched pages reach `extract` / `build_profile`).
3. **Output PII control** — detect/redact sensitive PII in the final profile and
   in audit logs.
4. **Compliance/audit** — log every guardrail decision for GDPR/CCPA audit and
   abuse tracing.

## Non-goals

- Replacing the LLM-based profile extraction with GLiNER2 (extraction stays as
  is; GLiNER2 is used only for PII NER + intent classification).
- Adopting a third-party guard framework (NeMo Guardrails / Guardrails-AI).
- Scanning short search snippets by default (only full fetched pages are
  scanned; snippet scanning is behind a flag, off by default).

## Architecture

Pluggable layered module that mirrors the project's existing multi-provider
style (LLM/search providers selected via config). All callers go through a
single cached facade `get_guardrails()` (like `get_settings`); models load
lazily on first use.

```
app/guardrails/
  __init__.py        # public facade: get_guardrails() -> Guardrails
  types.py           # GuardCategory, GuardAction, GuardFinding, GuardVerdict
  config.py          # GuardrailsSettings (nested into app.config.Settings)
  backends/
    base.py          # GuardBackend Protocol
    local.py         # in-process: gliner2 + gliner-guard-omni via transformers
    http.py          # sidecar client (httpx)
    noop.py          # disabled / no torch — always allow
  detectors.py       # SafetyDetector, PIIDetector, IntentDetector
  policy.py          # findings -> verdict per config
  engine.py          # Guardrails: check_input / scan_content / redact_profile + audit
  audit.py           # SQLite guard_events writer
```

### Layers

- **Backend** (`GuardBackend` Protocol) — raw engine, knows nothing about
  policy/categories. Two generic operations:
  - `async classify(text: str, labels: list[str]) -> dict[str, float]` — label
    scores (guard-omni + GLiNER2 classification).
  - `async extract(text: str, entity_types: list[str]) -> list[Span]` — NER
    spans (GLiNER2).
  Implementations: `local`, `http`, `noop`. Interchangeable — policy tests run
  on a fake backend without torch.
- **Detectors** — translate backend output into `GuardFinding`s:
  - `SafetyDetector` (guard-omni) → `jailbreak`, `prompt_injection`,
    `harmful_intent`, `toxicity`.
  - `PIIDetector` (GLiNER2 NER) → `pii` findings with entity-type labels
    (email, phone, address, id, …) and spans.
  - `IntentDetector` (GLiNER2 zero-shot classification) → OSINT intent:
    legitimate / stalking / doxing / minor (`harmful_intent`, `minor_target`).
- **Policy** (`policy.py`) — pure function mapping findings → verdict by config.
- **Engine** (`engine.py`) — facade with `check_input`, `scan_content`,
  `redact_profile`; writes audit.
- **Audit** (`audit.py`) — one row per fired finding into SQLite.

### Core types (`types.py`)

```python
GuardCategory = Literal[
    "jailbreak", "prompt_injection",   # safety (guard-omni)
    "harmful_intent",                  # stalking/doxing/harassment (guard-omni + intent)
    "minor_target",                    # searching for minors (intent)
    "toxicity",                        # guard-omni
    "pii",                             # GLiNER2 NER
]
GuardAction = Literal["allow", "block", "sanitize", "redact", "flag"]

@dataclass
class GuardFinding:
    category: GuardCategory
    score: float
    label: str                    # model label / entity type (email, phone…)
    spans: list[tuple[int, int]]  # char ranges (for sanitize/redact)

@dataclass
class GuardVerdict:
    action: GuardAction           # aggregated worst-case action per policy
    findings: list[GuardFinding]
    transformed_text: str | None  # sanitize/redact result; else None
    reason: str
```

## Configuration

`GuardrailsSettings` nests into `app.config.Settings` (env vars prefixed
`GUARDRAILS_`, CSV parsing in the project style).

```python
class GuardrailsSettings(BaseSettings):
    enabled: bool = True                       # GUARDRAILS_ENABLED
    backend: Literal["local", "http", "noop"] = "local"
    # models (for local backend)
    safety_model: str = "hivetrace/gliner-guard-omni"
    pii_model: str = "fastino/gliner2-..."     # exact id confirmed at impl time
    device: Literal["cpu", "cuda", "auto"] = "auto"
    http_url: str | None = None                # for backend=http
    # per-category policy: (action, threshold)
    policy: dict[GuardCategory, tuple[GuardAction, float]] = <defaults below>
    # active checkpoints (lets the expensive content scan be disabled)
    check_input: bool = True
    scan_content: bool = True
    scan_snippets: bool = False                # off by default; full pages only
    redact_output: bool = True
    fail_mode: Literal["open", "closed"] = "open"
```

### Default policy

| category         | action   | threshold |
|------------------|----------|-----------|
| harmful_intent   | block    | 0.6       |
| minor_target     | block    | 0.5       |
| jailbreak        | block    | 0.7       |
| prompt_injection | sanitize | 0.6       |
| toxicity         | flag     | 0.7       |
| pii              | redact   | 0.5       |

`policy.apply(findings, policy) -> GuardVerdict`:
- drop findings below their category threshold;
- aggregate to one action by priority `block > sanitize > redact > flag > allow`;
- for `sanitize`/`redact`, build `transformed_text` by replacing spans
  (injection spans → `[removed]`, PII spans → `[redacted:<type>]`).

Fully testable without models: feed `GuardFinding`s, assert the verdict.

## Integration points & data flow

All calls go through `get_guardrails()`. Models load lazily on first use.

**1. User input — inside graph nodes (covers Chainlit and A2A at once):**
- `collect_identity`: after resolving `first_name/last_name` (from start state and
  from the resume payload) → `check_input(full_query_text)` with Safety + Intent
  detectors.
- `narrow_query`: the resume answer (`value`/`extra`/free-text) → same
  `check_input`.
- On `action="block"` the node does **not** raise; it writes
  `guard_block: GuardVerdict` into state and sets `phase="blocked"`. A conditional
  edge to `END` is added from both nodes (and `route_after_*` honors `blocked`).
  Both frontends render the refusal from state — uniformly.

**2. Untrusted content before LLM — in `fetch_pages`:**
- inside `_fetch_and_extract`, before `extract_profile_from_page`:
  `scan_content(result.markdown)` (prompt_injection detector only — cheaper, no
  PII/intent). On `sanitize`, the `transformed_text` is passed to `extract`.
- snippet scanning in `preliminary_search` / `expand_search` is gated by
  `scan_snippets` (default off).

**3. Final output — in `build_profile`:**
- after assembling `PersonProfile` → `redact_profile(profile)`: serialize text
  fields, run the PII detector, replace spans with `[redacted:<type>]`, reassemble.
  One pass here covers both the Chainlit card and the A2A artifact (both read
  `state["profile"]`).

**Flow (happy path, guardrails on):**
```
input → collect_identity[check_input] → preliminary_search
      → … → fetch_pages[scan_content per page] → build_profile[redact_profile]
      → confirm_profile → END
block:  collect_identity / narrow_query [check_input → block] → END (refusal from state)
```

Blocking is implemented via **state + routing, not raise**, so LangGraph
checkpoints correctly and both frontends surface refusals the same way.

## Audit

New SQLite table `guard_events` in `data/app.db` (same style as `db/tokens.py`):

```
guard_events(
  id, ts, user_id, thread_id,
  point TEXT,            -- input | content | output
  category TEXT, action TEXT, score REAL,
  label TEXT, snippet TEXT,
  decision TEXT          -- allow/block/sanitize/redact/flag
)
```

- One row per fired finding.
- `snippet` is truncated (~200 chars) **and** itself run through PII-redaction
  before insert, so the audit log cannot become a PII leak.

## Error handling

- Backend crash/timeout → caught in `engine`; with `fail_mode="open"` return
  `allow` and write an audit event `action="flag", category="<backend_error>"`.
  The search is not interrupted.
- `enabled=False` or `backend="noop"` → facade returns `NoOpGuardrails`, zero
  overhead, nodes behave as before.
- Model load is lazy and once per process, with an explicit startup error if
  `backend="local"` but torch/weights are unavailable.

## Dependencies

- `gliner2` + `transformers`/`torch` move into the main `dependencies` (base
  install is now full).
- Weights are pulled from HuggingFace on first start — document in README, with a
  pre-download step option in the Dockerfile.

## Testing

pytest, following the repo conventions:

- `policy.py` — findings → expected verdict / `transformed_text` table-driven
  tests (no models, pure dataclasses).
- `engine` with a **FakeBackend** (deterministic `classify`/`extract`) — exercise
  all three points: block routing in the graph, content sanitize, profile redact.
- Integration: graph run with FakeBackend on an abuse request → reaches `END`
  with a refusal; injection in a page → `extract` receives cleaned text; PII in a
  profile → `[redacted:*]`.
- Real models are **not** loaded in CI (slow/heavy) — gated behind an opt-in
  `@pytest.mark.models` marker.

## Open items (resolve at implementation time)

- Exact GLiNER2 model id / package import surface and the label taxonomy each
  model emits (map model labels → `GuardCategory`).
- Confirm `gliner-guard-omni`'s output format (label set + score semantics).
- Whether `http` sidecar lands in this iteration or is stubbed (interface only).
