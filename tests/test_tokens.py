"""Tests for per-user API token storage."""

from __future__ import annotations

import pytest

from app.db.connection import init_db
from app.db.tokens import create_token, resolve_token, revoke_token
from app.db.users import create_user


@pytest.mark.asyncio
async def test_create_and_resolve_token():
    await init_db()
    user = await create_user("alice", "pw")
    plaintext = await create_token(user.id, label="ci")
    assert isinstance(plaintext, str) and len(plaintext) >= 32
    assert await resolve_token(plaintext) == user.id


@pytest.mark.asyncio
async def test_resolve_unknown_token_returns_none():
    await init_db()
    assert await resolve_token("nope-not-a-real-token") is None


@pytest.mark.asyncio
async def test_revoked_token_does_not_resolve():
    await init_db()
    user = await create_user("bob", "pw")
    plaintext = await create_token(user.id, label="tmp")
    assert await resolve_token(plaintext) == user.id
    await revoke_token(plaintext)
    assert await resolve_token(plaintext) is None


@pytest.mark.asyncio
async def test_token_hash_is_not_plaintext():
    await init_db()
    user = await create_user("carol", "pw")
    plaintext = await create_token(user.id)
    from app.db.connection import connect

    async with connect() as conn:
        row = await (
            await conn.execute("SELECT token_hash FROM api_tokens")
        ).fetchone()
    assert row["token_hash"] != plaintext
