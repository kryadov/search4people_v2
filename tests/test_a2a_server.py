"""Smoke tests for A2A config + server wiring."""

from __future__ import annotations


def test_a2a_config_defaults(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.a2a_port == 8001
    assert s.a2a_host == "0.0.0.0"
    assert s.a2a_public_url is None
