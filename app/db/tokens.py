"""Per-user API token storage for the A2A server.

Tokens are high-entropy random strings; we store only their sha256 hash so the
plaintext is never persisted. sha256 (not bcrypt) is used deliberately: lookup
must be a deterministic equality search, and the token's entropy makes a fast
hash safe here (unlike a low-entropy password).
"""

from __future__ import annotations

import hashlib
import secrets

from app.db.connection import connect


def _hash_token(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


async def create_token(user_id: int, label: str | None = None) -> str:
    """Create a token for a user and return the plaintext (shown once)."""
    plaintext = secrets.token_urlsafe(32)
    token_hash = _hash_token(plaintext)
    async with connect() as conn:
        await conn.execute(
            "INSERT INTO api_tokens (user_id, token_hash, label) VALUES (?, ?, ?)",
            (user_id, token_hash, label),
        )
        await conn.commit()
    return plaintext


async def resolve_token(plaintext: str) -> int | None:
    """Return the owning user_id for a live (non-revoked) token, else None."""
    token_hash = _hash_token(plaintext)
    async with connect() as conn:
        row = await (
            await conn.execute(
                "SELECT user_id FROM api_tokens "
                "WHERE token_hash = ? AND revoked_at IS NULL",
                (token_hash,),
            )
        ).fetchone()
    return int(row["user_id"]) if row else None


async def revoke_token(plaintext: str) -> bool:
    """Mark a token revoked. Returns True if a live token was revoked."""
    token_hash = _hash_token(plaintext)
    async with connect() as conn:
        cursor = await conn.execute(
            "UPDATE api_tokens SET revoked_at = datetime('now') "
            "WHERE token_hash = ? AND revoked_at IS NULL",
            (token_hash,),
        )
        await conn.commit()
        return cursor.rowcount > 0
