from app.config import Settings
from app.guardrails.config import DEFAULT_POLICY, GuardrailsSettings


def test_defaults_enabled_local():
    g = GuardrailsSettings()
    assert g.enabled is True
    assert g.backend == "local"
    assert g.fail_mode == "open"
    assert g.scan_snippets is False


def test_default_policy_present():
    assert GuardrailsSettings().policy["pii"] == ("redact", 0.5)
    assert DEFAULT_POLICY["harmful_intent"] == ("block", 0.6)


def test_nested_into_settings():
    s = Settings(_env_file=None)
    assert s.guardrails.backend == "local"
