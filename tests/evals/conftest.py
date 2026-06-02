"""Fixtures for DeepEval evals.

- Opt out of DeepEval network telemetry.
- Skip cleanly when the judge LLM is unavailable (cloud provider without a key,
  or local Ollama not reachable) instead of failing.
"""

from __future__ import annotations

import os

import pytest

# DeepEval's default per-task / gather timeouts assume a fast hosted judge. A
# local Ollama (e.g. gpt-oss:20b) is much slower, so raise the ceilings before
# DeepEval reads them. Set at import time so they apply to every eval session.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("ERROR_REPORTING", "0")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1200")
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "1200")


def _judge_unavailable_reason() -> str | None:
    """Return a human reason if the judge can't run, else None."""
    from app.config import get_settings

    settings = get_settings()
    provider = settings.llm_provider
    if provider == "anthropic" and not settings.anthropic_api_key:
        return "ANTHROPIC_API_KEY is not set"
    if provider == "openai" and not settings.openai_api_key:
        return "OPENAI_API_KEY is not set"
    if provider == "ollama":
        import httpx

        try:
            httpx.get(settings.ollama_base_url, timeout=2.0)
        except Exception:
            return f"Ollama not reachable at {settings.ollama_base_url}"
    return None


@pytest.fixture(scope="session", autouse=True)
def _deepeval_telemetry_off():
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    os.environ.setdefault("ERROR_REPORTING", "0")
    yield


@pytest.fixture(autouse=True)
def _require_judge() -> None:
    reason = _judge_unavailable_reason()
    if reason:
        pytest.skip(f"DeepEval judge unavailable: {reason}")
