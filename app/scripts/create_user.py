"""CLI helper to create a local user.

Usage:
    uv run s4p-create-user <username> <password> [--locale en|ru]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.db.connection import init_db
from app.db.users import create_user, get_user_by_username


async def _run(username: str, password: str, locale: str) -> int:
    await init_db()
    if await get_user_by_username(username) is not None:
        print(f"User '{username}' already exists.", file=sys.stderr)
        return 1
    user = await create_user(username, password, locale=locale)
    print(f"Created user '{user.username}' (id={user.id}, locale={user.locale}).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local search4people user.")
    parser.add_argument("username")
    parser.add_argument("password")
    parser.add_argument("--locale", choices=["en", "ru"], default="en")
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.username, args.password, args.locale)))


if __name__ == "__main__":
    main()
