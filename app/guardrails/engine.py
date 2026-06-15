"""Facade applying detectors + policy at each guardrail point, with audit."""

from __future__ import annotations

import structlog

from app.guardrails.audit import record_events
from app.guardrails.backends.base import GuardBackend
from app.guardrails.config import GuardrailsSettings
from app.guardrails.detectors import IntentDetector, PIIDetector, SafetyDetector
from app.guardrails.policy import apply
from app.guardrails.types import GuardVerdict

log = structlog.get_logger()

# Profile string fields that are free text and may carry stray PII. Structured
# fields (names, orgs, links) are intentionally left intact.
_REDACT_FIELDS = ("bio",)


class Guardrails:
    def __init__(self, backend: GuardBackend, settings: GuardrailsSettings) -> None:
        self._backend = backend
        self._settings = settings
        self._safety = SafetyDetector(backend)
        self._pii = PIIDetector(backend)
        self._intent = IntentDetector(backend)

    async def check_input(
        self, text: str, *, thread_id: str | None = None, user_id: int | None = None
    ) -> GuardVerdict:
        if not self._settings.check_input or not text.strip():
            return GuardVerdict(action="allow")
        try:
            findings = [
                *await self._safety.detect(text),
                *await self._intent.detect(text),
            ]
        except Exception as exc:
            return self._on_error("input", exc)
        verdict = apply(findings, self._settings.policy, text)
        await record_events(
            verdict,
            point="input",
            snippet_source=text,
            thread_id=thread_id,
            user_id=user_id,
        )
        return verdict

    async def scan_content(
        self, text: str, *, thread_id: str | None = None
    ) -> GuardVerdict:
        if not self._settings.scan_content or not text.strip():
            return GuardVerdict(action="allow")
        try:
            findings = await self._safety.detect(text, injection_only=True)
        except Exception as exc:
            return self._on_error("content", exc)
        verdict = apply(findings, self._settings.policy, text)
        await record_events(
            verdict, point="content", snippet_source=text, thread_id=thread_id
        )
        return verdict

    async def redact_profile(
        self, profile: dict, *, thread_id: str | None = None
    ) -> tuple[dict, GuardVerdict]:
        if not self._settings.redact_output:
            return profile, GuardVerdict(action="allow")
        out = dict(profile)
        all_findings: list = []
        worst = GuardVerdict(action="allow")
        for field in _REDACT_FIELDS:
            value = out.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            try:
                findings = await self._pii.detect(value)
            except Exception as exc:
                return out, self._on_error("output", exc)
            verdict = apply(findings, self._settings.policy, value)
            if verdict.transformed_text is not None:
                out[field] = verdict.transformed_text
            all_findings.extend(verdict.findings)
            if verdict.action != "allow":
                worst = verdict
        if all_findings:
            await record_events(
                worst, point="output", snippet_source="<profile>", thread_id=thread_id
            )
        return out, worst

    def _on_error(self, point: str, exc: Exception) -> GuardVerdict:
        log.warning("guard_backend_error", point=point, error=str(exc))
        if self._settings.fail_mode == "closed":
            return GuardVerdict(action="block", reason="guardrail backend unavailable")
        return GuardVerdict(action="allow", reason="backend_error")


class NoOpGuardrails:
    async def check_input(self, text: str, **_: object) -> GuardVerdict:
        return GuardVerdict(action="allow")

    async def scan_content(self, text: str, **_: object) -> GuardVerdict:
        return GuardVerdict(action="allow")

    async def redact_profile(self, profile: dict, **_: object) -> tuple[dict, GuardVerdict]:
        return profile, GuardVerdict(action="allow")
