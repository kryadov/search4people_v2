"""Guardrails configuration (nested into app.config.Settings)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.guardrails.types import GuardAction, GuardCategory

# (action, threshold) per category. Findings below threshold are dropped.
DEFAULT_POLICY: dict[GuardCategory, tuple[GuardAction, float]] = {
    "harmful_intent": ("block", 0.6),
    "minor_target": ("block", 0.5),
    "jailbreak": ("block", 0.7),
    "prompt_injection": ("sanitize", 0.6),
    "toxicity": ("flag", 0.7),
    "pii": ("redact", 0.5),
}


class GuardrailsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="guardrails_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = True
    backend: Literal["local", "http", "noop"] = "local"

    safety_model: str = "hivetrace/gliner-guard-omni"
    pii_model: str = "fastino/gliner2-base-v1"
    device: Literal["cpu", "cuda", "auto"] = "auto"
    http_url: str | None = None

    check_input: bool = True
    scan_content: bool = True
    scan_snippets: bool = False
    redact_output: bool = True
    fail_mode: Literal["open", "closed"] = "open"

    policy: dict[GuardCategory, tuple[GuardAction, float]] = Field(
        default_factory=lambda: dict(DEFAULT_POLICY)
    )
