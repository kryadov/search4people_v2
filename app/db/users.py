"""User table accessors."""

from __future__ import annotations

from dataclasses import dataclass

import bcrypt

from app.db.connection import connect


@dataclass(slots=True)
class User:
    id: int
    username: str
    locale: str


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


async def create_user(username: str, password: str, locale: str = "en") -> User:
    password_hash = _hash_password(password)
    async with connect() as conn:
        cursor = await conn.execute(
            "INSERT INTO users (username, password_hash, locale) VALUES (?, ?, ?)",
            (username, password_hash, locale),
        )
        await conn.commit()
        return User(id=cursor.lastrowid or 0, username=username, locale=locale)


async def authenticate(username: str, password: str) -> User | None:
    async with connect() as conn:
        row = await (
            await conn.execute(
                "SELECT id, username, password_hash, locale FROM users WHERE username = ?",
                (username,),
            )
        ).fetchone()
    if row is None:
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    return User(id=row["id"], username=row["username"], locale=row["locale"])


async def set_user_locale(user_id: int, locale: str) -> None:
    async with connect() as conn:
        await conn.execute("UPDATE users SET locale = ? WHERE id = ?", (locale, user_id))
        await conn.commit()


async def get_user_by_username(username: str) -> User | None:
    async with connect() as conn:
        row = await (
            await conn.execute(
                "SELECT id, username, locale FROM users WHERE username = ?",
                (username,),
            )
        ).fetchone()
    if row is None:
        return None
    return User(id=row["id"], username=row["username"], locale=row["locale"])
