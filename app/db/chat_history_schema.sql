-- Chainlit data-layer schema (SQLite). Managed by us because
-- SQLAlchemyDataLayer does not create its own tables. Kept in a separate DB
-- file from app.db: Chainlit's `users` table collides with our auth `users`.

CREATE TABLE IF NOT EXISTS users (
    "id"         TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata"   TEXT NOT NULL,
    "createdAt"  TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id"             TEXT PRIMARY KEY,
    "createdAt"      TEXT,
    "name"           TEXT,
    "userId"         TEXT,
    "userIdentifier" TEXT,
    "tags"           TEXT,
    "metadata"       TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id"            TEXT PRIMARY KEY,
    "name"          TEXT NOT NULL,
    "type"          TEXT NOT NULL,
    "threadId"      TEXT NOT NULL,
    "parentId"      TEXT,
    "streaming"     INTEGER NOT NULL,
    "waitForAnswer" INTEGER,
    "isError"       INTEGER,
    "metadata"      TEXT,
    "tags"          TEXT,
    "input"         TEXT,
    "output"        TEXT,
    "createdAt"     TEXT,
    "command"       TEXT,
    "start"         TEXT,
    "end"           TEXT,
    "generation"    TEXT,
    "showInput"     TEXT,
    "language"      TEXT,
    "indent"        INTEGER,
    "defaultOpen"   INTEGER,
    "autoCollapse"  INTEGER,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id"           TEXT PRIMARY KEY,
    "threadId"     TEXT,
    "type"         TEXT,
    "url"          TEXT,
    "chainlitKey"  TEXT,
    "name"         TEXT NOT NULL,
    "display"      TEXT,
    "objectKey"    TEXT,
    "size"         TEXT,
    "page"         INTEGER,
    "language"     TEXT,
    "forId"        TEXT,
    "mime"         TEXT,
    "props"        TEXT,
    "autoPlay"     INTEGER,
    "playerConfig" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id"       TEXT PRIMARY KEY,
    "forId"    TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value"    INTEGER NOT NULL,
    "comment"  TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

-- get_all_user_threads (the sidebar listing) filters on threads."userId".
CREATE INDEX IF NOT EXISTS idx_threads_user ON threads("userId");
CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps("threadId");
CREATE INDEX IF NOT EXISTS idx_elements_thread ON elements("threadId");
CREATE INDEX IF NOT EXISTS idx_feedbacks_for ON feedbacks("forId");
