"""Tests for A2A Bearer-token auth middleware + contextvar."""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.a2a.auth import BearerAuthMiddleware, current_user_id
from app.db.connection import init_db
from app.db.tokens import create_token
from app.db.users import create_user


def _app() -> Starlette:
    async def whoami(request):
        return JSONResponse({"user_id": current_user_id()})

    async def card(request):
        return JSONResponse({"ok": True})

    app = Starlette(
        routes=[
            Route("/", whoami, methods=["POST"]),
            Route("/.well-known/agent-card.json", card, methods=["GET"]),
        ]
    )
    app.add_middleware(BearerAuthMiddleware)
    return app


@pytest.mark.asyncio
async def test_valid_token_sets_user_id():
    await init_db()
    user = await create_user("dave", "pw")
    token = await create_token(user.id)
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["user_id"] == user.id


@pytest.mark.asyncio
async def test_missing_token_rejected():
    await init_db()
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token_rejected():
    await init_db()
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post("/", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_agent_card_is_public():
    await init_db()
    transport = httpx.ASGITransport(app=_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.get("/.well-known/agent-card.json")
    assert r.status_code == 200
