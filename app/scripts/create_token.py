"""CLI helper to mint an A2A API token for an existing user.

Usage:
    uv run s4p-create-token <username> [--label LABEL]

Prints the plaintext token exactly once — store it now, it is not recoverable.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.db.connection import init_db
from app.db.tokens import create_token
from app.db.users import get_user_by_username


async def _run(username: str, label: str | None) -> int:
    await init_db()
    user = await get_user_by_username(username)
    if user is None:
        print(f"User '{username}' does not exist.", file=sys.stderr)
        return 1
    plaintext = await create_token(user.id, label=label)
    print(f"Token for '{username}' (id={user.id}) — store it now, shown once:")
    print(plaintext)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint an A2A API token.")
    parser.add_argument("username")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.username, args.label)))


if __name__ == "__main__":
    main()
