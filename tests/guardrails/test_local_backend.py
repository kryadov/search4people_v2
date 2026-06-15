import pytest

pytestmark = pytest.mark.models


async def test_local_backend_extracts_email():
    from app.guardrails.backends.local import LocalGlinerBackend
    from app.guardrails.config import GuardrailsSettings

    backend = LocalGlinerBackend(GuardrailsSettings(device="cpu"))
    spans = await backend.extract("write to john@example.com", ["email"])
    assert any(s.label == "email" for s in spans)
