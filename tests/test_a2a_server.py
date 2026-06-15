"""Smoke tests for A2A config + server wiring."""

from __future__ import annotations

import pytest


def test_a2a_config_defaults(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.a2a_port == 8001
    assert s.a2a_host == "0.0.0.0"
    assert s.a2a_public_url is None


def test_build_agent_card_shape():
    from app.a2a.server import build_agent_card

    card = build_agent_card()
    assert card.name == "search4people"
    skill_ids = [s.id for s in card.skills]
    assert "people_search" in skill_ids
    # Bearer security scheme is declared.
    assert "bearer" in (card.security_schemes or {})


def test_build_agent_card_uses_public_url(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("A2A_PUBLIC_URL", "https://example.test/")
    get_settings.cache_clear()
    from app.a2a.server import build_agent_card

    card = build_agent_card()
    assert str(card.url).startswith("https://example.test")
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_agent_card_served_and_auth_enforced(monkeypatch):
    import httpx

    from app.a2a.server import build_app

    app = await build_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        card = await c.get("/.well-known/agent-card.json")
        assert card.status_code == 200
        body = card.json()
        assert body["name"] == "search4people"

        rpc = await c.post(
            "/",
            json={"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {}},
        )
        assert rpc.status_code == 401
