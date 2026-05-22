"""Chainlit-facing auth glue."""

from __future__ import annotations

import chainlit as cl

from app.db.users import authenticate


@cl.password_auth_callback
async def password_auth(username: str, password: str) -> cl.User | None:
    """Validate username/password against the local SQLite users table."""
    user = await authenticate(username, password)
    if user is None:
        return None
    return cl.User(
        identifier=user.username,
        metadata={"id": user.id, "locale": user.locale},
    )
