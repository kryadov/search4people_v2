"""Bearer-token authentication for the A2A server.

A Starlette middleware validates `Authorization: Bearer <token>` against the
`api_tokens` table and stashes the resolved user_id in a contextvar that the
executor reads. Paths under `/.well-known/` are public (Agent Cards are meant
to be fetched without auth).
"""

from __future__ import annotations

import json
from contextvars import ContextVar

from starlette.types import ASGIApp, Receive, Scope, Send

from app.db.tokens import resolve_token

_current_user_id: ContextVar[int | None] = ContextVar("a2a_current_user_id", default=None)

_PUBLIC_PREFIXES = ("/.well-known/",)


def current_user_id() -> int | None:
    """The authenticated user_id for the in-flight A2A request, if any."""
    return _current_user_id.get()


class BearerAuthMiddleware:
    """Pure-ASGI Bearer-token middleware.

    Uses a raw ASGI ``__call__`` instead of ``BaseHTTPMiddleware`` so that the
    contextvar set here is visible to the downstream route handler running in
    the *same* asyncio Task context.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            await self.app(scope, receive, send)
            return

        # Extract the Authorization header from raw ASGI scope headers.
        headers: dict[bytes, bytes] = {
            k.lower(): v for k, v in scope.get("headers", [])
        }
        auth_header: str = headers.get(b"authorization", b"").decode("latin-1")

        if not auth_header.lower().startswith("bearer "):
            await _send_401(send, "missing bearer token")
            return

        token = auth_header[len("bearer "):].strip()
        user_id = await resolve_token(token)
        if user_id is None:
            await _send_401(send, "invalid token")
            return

        reset = _current_user_id.set(user_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _current_user_id.reset(reset)


async def _send_401(send: Send, detail: str) -> None:
    body = json.dumps({"error": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
