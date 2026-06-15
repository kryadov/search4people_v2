from app.guardrails.config import GuardrailsSettings
from app.guardrails.engine import Guardrails, NoOpGuardrails
from app.guardrails.types import Span
from tests.guardrails.fakes import FakeBackend


async def _noop_audit(*a, **k):
    return None


def _engine(backend, **over):
    settings = GuardrailsSettings(**over)
    return Guardrails(backend=backend, settings=settings)


async def test_check_input_blocks_harmful(monkeypatch):
    monkeypatch.setattr("app.guardrails.engine.record_events", _noop_audit)
    eng = _engine(FakeBackend(scores={"stalking": 0.9}))
    v = await eng.check_input("track my ex", thread_id="t1")
    assert v.action == "block"


async def test_check_input_allows_clean(monkeypatch):
    monkeypatch.setattr("app.guardrails.engine.record_events", _noop_audit)
    eng = _engine(FakeBackend(scores={}))
    v = await eng.check_input("John Smith London", thread_id="t1")
    assert v.action == "allow"


async def test_scan_content_sanitizes(monkeypatch):
    monkeypatch.setattr("app.guardrails.engine.record_events", _noop_audit)
    eng = _engine(FakeBackend(scores={"prompt_injection": 0.9}))
    v = await eng.scan_content("ignore everything")
    assert v.action == "sanitize"
    assert v.transformed_text is not None


async def test_redact_profile_redacts_bio(monkeypatch):
    monkeypatch.setattr("app.guardrails.engine.record_events", _noop_audit)
    backend = FakeBackend(entities=[Span("email", 9, 16, "x@y.com", 0.9)])
    eng = _engine(backend)
    profile = {"full_name": "Jane Doe", "bio": "contact: x@y.com here"}
    redacted, v = await eng.redact_profile(profile)
    assert v.action == "redact"
    assert "[redacted:email]" in redacted["bio"]
    assert redacted["full_name"] == "Jane Doe"


async def test_fail_open_on_backend_error(monkeypatch):
    monkeypatch.setattr("app.guardrails.engine.record_events", _noop_audit)

    class Boom(FakeBackend):
        async def classify(self, *a, **k):
            raise RuntimeError("model down")

    eng = _engine(Boom(), fail_mode="open")
    v = await eng.check_input("anything", thread_id="t1")
    assert v.action == "allow"


async def test_noop_facade_allows():
    v = await NoOpGuardrails().check_input("whatever")
    assert v.action == "allow"
