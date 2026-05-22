-- Application schema. LangGraph checkpoints live in their own tables managed
-- by langgraph-checkpoint-sqlite in the same database file.

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    locale        TEXT    NOT NULL DEFAULT 'en',
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS profiles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    thread_id     TEXT    NOT NULL,
    full_name     TEXT    NOT NULL,
    profile_json  TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_thread ON profiles(thread_id);

CREATE TABLE IF NOT EXISTS source_evidence (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id    INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    url           TEXT    NOT NULL,
    platform      TEXT,
    snippet       TEXT,
    captured_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_evidence_profile ON source_evidence(profile_id);
