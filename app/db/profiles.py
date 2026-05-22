"""Profile + source-evidence persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.db.connection import connect


@dataclass(slots=True)
class StoredProfile:
    id: int
    user_id: int
    thread_id: str
    full_name: str
    profile: dict[str, Any]
    created_at: str


async def save_profile(
    user_id: int,
    thread_id: str,
    full_name: str,
    profile: dict[str, Any],
    sources: list[dict[str, str | None]] | None = None,
) -> int:
    """Persist a built profile and its source evidence rows.

    Returns the new profile id.
    """
    async with connect() as conn:
        cursor = await conn.execute(
            "INSERT INTO profiles (user_id, thread_id, full_name, profile_json) "
            "VALUES (?, ?, ?, ?)",
            (user_id, thread_id, full_name, json.dumps(profile, ensure_ascii=False)),
        )
        profile_id = cursor.lastrowid or 0
        if sources:
            await conn.executemany(
                "INSERT INTO source_evidence (profile_id, url, platform, snippet) "
                "VALUES (?, ?, ?, ?)",
                [
                    (profile_id, s.get("url"), s.get("platform"), s.get("snippet"))
                    for s in sources
                    if s.get("url")
                ],
            )
        await conn.commit()
        return profile_id


async def list_profiles(user_id: int, limit: int = 50) -> list[StoredProfile]:
    async with connect() as conn:
        rows = await (
            await conn.execute(
                "SELECT id, user_id, thread_id, full_name, profile_json, created_at "
                "FROM profiles WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
        ).fetchall()
    return [
        StoredProfile(
            id=row["id"],
            user_id=row["user_id"],
            thread_id=row["thread_id"],
            full_name=row["full_name"],
            profile=json.loads(row["profile_json"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]
