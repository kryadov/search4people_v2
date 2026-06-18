# Persistent Per-User Chat History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each authenticated Chainlit user a persistent, browsable, resumable chat history with built-in search and automatic per-search tags.

**Architecture:** Register Chainlit's official `SQLAlchemyDataLayer` against a dedicated `data/chat_history.db` (separate from `data/app.db` to avoid the `users` table-name collision). Make Chainlit's persisted thread id the graph `thread_id` so resuming a thread restores both the UI messages and the LangGraph checkpoint. Search is Chainlit's built-in substring search; auto-tags are computed on completion, stored in thread metadata, and echoed as a searchable line.

**Tech Stack:** Python 3.13, Chainlit ≥2.11.1, SQLAlchemy 2.x + greenlet (async SQLite via aiosqlite), LangGraph, pytest/pytest-asyncio.

## Global Constraints

- Python `>=3.13,<3.14`; new deps go in main `dependencies` (not a dev/optional group), enabled by default.
- New deps pinned: `sqlalchemy>=2.0,<3.0`, `greenlet>=3.0`. `aiosqlite` already present.
- SQLite only — no Postgres/LiteralAI. No FTS5, no profile-table search, no manual tags.
- Every new user-facing string must exist in both `en` and `ru` in `app/i18n.py`.
- Do not change graph topology, nodes, prompts, or guardrails. The A2A frontend is untouched.
- Chat history lives in a **separate** `data/chat_history.db`; `data/app.db` (auth users, profiles, checkpoints) is unchanged.
- The graph `thread_id` is unified with Chainlit's `cl.context.session.thread_id`.
- `uv run pytest`, `uv run ruff check .`, and `uv run mypy app` must stay green.
- Known constraint: Chainlit's `update_thread(tags=[...])` binds a Python list, which SQLite rejects — so canonical tags are written via `update_thread(metadata={"auto_tags": [...]})` (a JSON string bind), never via the `tags=` parameter.

---

### Task 1: Dependencies, config setting, and test env

**Files:**
- Modify: `pyproject.toml:10-43` (add deps)
- Modify: `app/config.py:77-78` (add `chat_history_db_path`)
- Modify: `.env.example` (document the new path)
- Modify: `tests/conftest.py:28-33` (isolate the new DB path in tests)
- Test: `tests/test_chat_history_config.py`

**Interfaces:**
- Produces: `Settings.chat_history_db_path: Path` (default `Path("data/chat_history.db")`), overridable via env `CHAT_HISTORY_DB_PATH`.

- [ ] **Step 1: Add dependencies to `pyproject.toml`**

In the `dependencies` list, under the `# Auth + storage` group, change:

```toml
    # Auth + storage
    "bcrypt>=4.2.0",
    "aiosqlite>=0.20.0",
```

to:

```toml
    # Auth + storage
    "bcrypt>=4.2.0",
    "aiosqlite>=0.20.0",
    # Chainlit data layer (persistent chat history)
    "sqlalchemy>=2.0,<3.0",
    "greenlet>=3.0",
```

- [ ] **Step 2: Install**

Run: `uv sync`
Expected: resolves and installs `sqlalchemy` and `greenlet`.

- [ ] **Step 3: Add the config setting**

In `app/config.py`, in the `# Storage` block, change:

```python
    # Storage
    db_path: Path = Path("data/app.db")
```

to:

```python
    # Storage
    db_path: Path = Path("data/app.db")
    # Persistent Chainlit chat history (separate file: Chainlit's data-layer
    # schema defines its own `users` table, which collides with our auth table).
    chat_history_db_path: Path = Path("data/chat_history.db")
```

- [ ] **Step 4: Document in `.env.example`**

Find the storage section in `.env.example` (the line with `DB_PATH=`) and add directly beneath it:

```bash
# Persistent chat history (Chainlit data layer). Separate file from DB_PATH.
CHAT_HISTORY_DB_PATH=data/chat_history.db
```

- [ ] **Step 5: Isolate the new DB path in tests**

In `tests/conftest.py`, in the `_isolate_env` fixture, after the `monkeypatch.setenv("DB_PATH", ...)` line add:

```python
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(tmp_path / "chat_history.db"))
```

- [ ] **Step 6: Write the failing test**

Create `tests/test_chat_history_config.py`:

```python
"""Config + dependency wiring for persistent chat history."""

from __future__ import annotations

from pathlib import Path


def test_chat_history_db_path_default(monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.delenv("CHAT_HISTORY_DB_PATH", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.chat_history_db_path == Path("data/chat_history.db")


def test_chat_history_db_path_env_override(monkeypatch) -> None:
    from app.config import get_settings

    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", "/tmp/custom_history.db")
    get_settings.cache_clear()
    assert get_settings().chat_history_db_path == Path("/tmp/custom_history.db")


def test_sqlalchemy_importable() -> None:
    import sqlalchemy  # noqa: F401
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: F401
```

- [ ] **Step 7: Run the test**

Run: `uv run pytest tests/test_chat_history_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock app/config.py .env.example tests/conftest.py tests/test_chat_history_config.py
git commit -m "feat(history): add chat_history_db_path setting + sqlalchemy/greenlet deps"
```

---

### Task 2: Chainlit SQLite schema + `init_chat_history_db`

**Files:**
- Create: `app/db/chat_history_schema.sql`
- Create: `app/db/chat_history.py`
- Test: `tests/test_chat_history_db.py`

**Interfaces:**
- Produces: `app.db.chat_history.init_chat_history_db() -> None` — idempotent; creates the five Chainlit tables in `settings.chat_history_db_path` and sets WAL mode.

- [ ] **Step 1: Create the schema file**

Create `app/db/chat_history_schema.sql` (Chainlit's data-layer schema, SQLite-adapted; the `tags` columns are plain `TEXT` because we never bind list values — see plan constraints):

```sql
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

CREATE INDEX IF NOT EXISTS idx_threads_user ON threads("userIdentifier");
CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps("threadId");
CREATE INDEX IF NOT EXISTS idx_elements_thread ON elements("threadId");
CREATE INDEX IF NOT EXISTS idx_feedbacks_for ON feedbacks("forId");
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_chat_history_db.py`:

```python
"""Chat-history DB init + data-layer factory."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest


@pytest.mark.anyio
async def test_init_chat_history_db_creates_tables(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db))
    from app.config import get_settings

    get_settings.cache_clear()

    from app.db.chat_history import init_chat_history_db

    await init_chat_history_db()
    await init_chat_history_db()  # idempotent: second call must not raise

    async with aiosqlite.connect(db) as conn:
        rows = await (
            await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        mode = await (await conn.execute("PRAGMA journal_mode")).fetchone()

    names = {r[0] for r in rows}
    assert {"users", "threads", "steps", "elements", "feedbacks"} <= names
    assert mode is not None and mode[0].lower() == "wal"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_chat_history_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.db.chat_history'`.

- [ ] **Step 4: Implement `app/db/chat_history.py`**

Create `app/db/chat_history.py`:

```python
"""Persistent chat-history storage: Chainlit data-layer DB + factory.

Lives in its own SQLite file (`settings.chat_history_db_path`) because
Chainlit's data-layer schema defines a `users` table that collides with the
auth `users` table in `app.db`.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from app.config import get_settings

_SCHEMA_PATH = Path(__file__).with_name("chat_history_schema.sql")


async def init_chat_history_db() -> None:
    """Create the chat-history DB (if missing), apply the schema, enable WAL."""
    settings = get_settings()
    settings.chat_history_db_path.parent.mkdir(parents=True, exist_ok=True)
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    async with aiosqlite.connect(settings.chat_history_db_path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.executescript(sql)
        await conn.commit()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_chat_history_db.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/db/chat_history_schema.sql app/db/chat_history.py tests/test_chat_history_db.py
git commit -m "feat(history): chat-history SQLite schema + idempotent init"
```

---

### Task 3: Data-layer factory `build_data_layer`

**Files:**
- Modify: `app/db/chat_history.py` (add factory)
- Test: `tests/test_chat_history_db.py` (add a case)

**Interfaces:**
- Consumes: `Settings.chat_history_db_path`.
- Produces: `app.db.chat_history.build_data_layer() -> SQLAlchemyDataLayer` — async SQLite conninfo, `storage_provider=None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chat_history_db.py`:

```python
def test_build_data_layer_uses_sqlite_conninfo(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "history.db"
    monkeypatch.setenv("CHAT_HISTORY_DB_PATH", str(db))
    from app.config import get_settings

    get_settings.cache_clear()

    from app.db.chat_history import build_data_layer

    dl = build_data_layer()
    assert dl._conninfo.startswith("sqlite+aiosqlite:///")
    assert dl._conninfo.endswith("history.db")
    assert dl.storage_provider is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_chat_history_db.py::test_build_data_layer_uses_sqlite_conninfo -v`
Expected: FAIL with `ImportError: cannot import name 'build_data_layer'`.

- [ ] **Step 3: Implement the factory**

In `app/db/chat_history.py`, add the import at the top (after the existing imports):

```python
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
```

and append:

```python
def build_data_layer() -> SQLAlchemyDataLayer:
    """Construct the Chainlit data layer over the chat-history SQLite file.

    `as_posix()` keeps the URL valid on Windows (no backslashes). No storage
    provider: this app persists only markdown messages, no binary elements.
    """
    settings = get_settings()
    conninfo = f"sqlite+aiosqlite:///{settings.chat_history_db_path.as_posix()}"
    return SQLAlchemyDataLayer(conninfo=conninfo, storage_provider=None)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_chat_history_db.py -v`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add app/db/chat_history.py tests/test_chat_history_db.py
git commit -m "feat(history): SQLAlchemyDataLayer factory over chat_history.db"
```

---

### Task 4: Auto-tag computation + searchable tag line + i18n

**Files:**
- Create: `app/ui/tags.py`
- Modify: `app/i18n.py:113` (add `tags_line_label`)
- Test: `tests/test_tags.py`

**Interfaces:**
- Consumes: a `PersonProfile.model_dump(mode="json")` dict and a locale.
- Produces:
  - `app.ui.tags.compute_auto_tags(profile: dict[str, Any], locale: str) -> list[str]`
  - `app.ui.tags.render_tag_line(tags: list[str], locale: Locale) -> str`

- [ ] **Step 1: Add the i18n string**

In `app/i18n.py`, inside `_TRANSLATIONS`, before the closing `}` of the dict (after the `language_toggle_hint` entry) add:

```python
    "tags_line_label": {
        "en": "Tags:",
        "ru": "Теги:",
    },
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_tags.py`:

```python
"""Auto-tag computation + searchable tag-line rendering."""

from __future__ import annotations

from app.ui.tags import compute_auto_tags, render_tag_line


def _profile(**over):
    base = {
        "full_name": "Jane Doe",
        "confidence": "high",
        "evidence": [
            {"url": "https://github.com/jane", "platform": "github"},
            {"url": "https://linkedin.com/in/jane", "platform": "linkedin"},
            {"url": "https://github.com/jane2", "platform": "github"},  # dup
            {"url": "https://example.com/x", "platform": None},  # no platform
        ],
    }
    base.update(over)
    return base


def test_compute_auto_tags_dedups_platforms_and_adds_confidence_locale() -> None:
    tags = compute_auto_tags(_profile(), "ru")
    assert tags == [
        "platform:github",
        "platform:linkedin",
        "confidence:high",
        "locale:ru",
    ]


def test_compute_auto_tags_defaults_confidence_when_missing() -> None:
    tags = compute_auto_tags({"full_name": "X", "evidence": []}, "en")
    assert tags == ["confidence:medium", "locale:en"]


def test_render_tag_line_includes_label_and_tokens() -> None:
    line = render_tag_line(["platform:github", "confidence:high"], "en")
    assert "Tags:" in line
    assert "platform:github" in line
    assert "confidence:high" in line


def test_render_tag_line_is_localized() -> None:
    line = render_tag_line(["locale:ru"], "ru")
    assert "Теги:" in line
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/test_tags.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ui.tags'`.

- [ ] **Step 4: Implement `app/ui/tags.py`**

Create `app/ui/tags.py`:

```python
"""Automatic per-search thread tags.

Tags are namespaced tokens (`platform:github`, `confidence:high`, `locale:ru`)
so the built-in Chainlit search can find threads precisely. They are stored in
thread metadata and echoed as a searchable line in the conversation.
"""

from __future__ import annotations

from typing import Any

from app.i18n import Locale, t


def compute_auto_tags(profile: dict[str, Any], locale: str) -> list[str]:
    """Derive deduped, namespaced tags from a built profile + conversation locale."""
    tags: list[str] = []
    seen: set[str] = set()
    for ev in profile.get("evidence") or []:
        platform = (ev.get("platform") or "").strip().lower()
        if not platform:
            continue
        tag = f"platform:{platform}"
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    confidence = profile.get("confidence") or "medium"
    tags.append(f"confidence:{confidence}")
    tags.append(f"locale:{locale}")
    return tags


def render_tag_line(tags: list[str], locale: Locale) -> str:
    """A compact, searchable one-liner echoing the tags into the thread."""
    label = t("tags_line_label", locale)
    return f"🏷 {label} " + " · ".join(tags)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_tags.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add app/ui/tags.py app/i18n.py tests/test_tags.py
git commit -m "feat(history): auto-tag computation + searchable tag line"
```

---

### Task 5: Resume session-state derivation

**Files:**
- Create: `app/ui/session_state.py`
- Test: `tests/test_session_state.py`

**Interfaces:**
- Consumes: `app.graph.bridge.PendingInput` (fields `kind`, `attribute`, `candidate_count`).
- Produces:
  - `app.ui.session_state.SessionState` dataclass: `awaiting: str | None`, `narrowing_attribute: str | None = None`, `narrowing_candidate_count: int = 0`.
  - `app.ui.session_state.derive_session_state(pending: PendingInput | None) -> SessionState`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_session_state.py`:

```python
"""Restoring `awaiting` session state from a graph snapshot on resume."""

from __future__ import annotations

from app.graph.bridge import PendingInput
from app.ui.session_state import SessionState, derive_session_state


def test_none_falls_back_to_identity() -> None:
    # A resumed thread with no pending interrupt (e.g. done) → start fresh.
    assert derive_session_state(None) == SessionState(awaiting="identity")


def test_ask_identity() -> None:
    pending = PendingInput("ask_identity", "q", {}, 0, None)
    assert derive_session_state(pending) == SessionState(awaiting="identity")


def test_ask_narrowing_carries_attribute_and_count() -> None:
    pending = PendingInput("ask_narrowing", "q", {}, 3, "city")
    assert derive_session_state(pending) == SessionState(
        awaiting="narrowing", narrowing_attribute="city", narrowing_candidate_count=3
    )


def test_confirm_profile() -> None:
    pending = PendingInput("confirm_profile", "q", {}, 0, None)
    assert derive_session_state(pending) == SessionState(awaiting="confirm")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_session_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ui.session_state'`.

- [ ] **Step 3: Implement `app/ui/session_state.py`**

Create `app/ui/session_state.py`:

```python
"""Derive Chainlit session flags from the graph's pending interrupt.

On resume, `cl.user_session` is empty but the LangGraph checkpoint knows what
the conversation is waiting for. This pure mapping turns the checkpoint-derived
`PendingInput` into the `awaiting` flag (plus narrowing context) so the next
user reply routes correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.graph.bridge import PendingInput


@dataclass(frozen=True)
class SessionState:
    awaiting: str | None
    narrowing_attribute: str | None = None
    narrowing_candidate_count: int = 0


def derive_session_state(pending: PendingInput | None) -> SessionState:
    """Map a pending interrupt to the Chainlit `awaiting` session state.

    `None` (no interrupt — e.g. a finished or fresh thread) falls back to
    `identity` so a resumed thread is always usable.
    """
    if pending is None:
        return SessionState(awaiting="identity")
    if pending.kind == "ask_narrowing":
        return SessionState(
            awaiting="narrowing",
            narrowing_attribute=pending.attribute,
            narrowing_candidate_count=pending.candidate_count,
        )
    if pending.kind == "confirm_profile":
        return SessionState(awaiting="confirm")
    return SessionState(awaiting="identity")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_session_state.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/ui/session_state.py tests/test_session_state.py
git commit -m "feat(history): derive awaiting state from graph snapshot for resume"
```

---

### Task 6: `fresh_search_input` for follow-up-after-done

**Files:**
- Modify: `app/graph/bridge.py` (add function)
- Test: `tests/test_bridge.py` (add a case)

**Interfaces:**
- Consumes: a parsed identity dict (from `parse_identity_text`) and a `Locale`.
- Produces: `app.graph.bridge.fresh_search_input(identity: dict[str, Any], locale: Locale) -> dict[str, Any]` — a full `PeopleSearchState` patch that resets all transient channels for a new search while preserving the message history (the `messages` channel is intentionally omitted, as it uses the `add_messages` reducer).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bridge.py`:

```python
def test_fresh_search_input_resets_transient_state() -> None:
    from app.graph.bridge import fresh_search_input

    out = fresh_search_input({"first_name": "Jane", "last_name": "Doe"}, "ru")
    assert out["query"] == {"first_name": "Jane", "last_name": "Doe"}
    assert out["locale"] == "ru"
    assert out["phase"] == "collect"
    assert out["candidates"] == []
    assert out["fetched_pages"] == []
    assert out["visited_urls"] == []
    assert out["iteration"] == 0
    assert out["profile"] is None
    assert out["user_decision"] is None
    assert out["selected_candidate_index"] is None
    assert out["guard_block"] is None
    # messages must NOT be reset (add_messages reducer preserves history).
    assert "messages" not in out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_bridge.py::test_fresh_search_input_resets_transient_state -v`
Expected: FAIL with `ImportError: cannot import name 'fresh_search_input'`.

- [ ] **Step 3: Implement the function**

In `app/graph/bridge.py`, add after `parse_identity_text` (before `read_pending_input`):

```python
def fresh_search_input(identity: dict[str, Any], locale: Locale) -> dict[str, Any]:
    """A full state reset for a brand-new search on an existing thread.

    Used when a finished (`phase == "done"`) thread receives a new identity:
    the conversation continues in the same thread, but every plain (non-reduced)
    state channel is overwritten so the new search does not inherit stale
    candidates/profile. `messages` is omitted on purpose — its `add_messages`
    reducer must keep the prior history visible.
    """
    return {
        "query": identity,
        "locale": locale,
        "candidates": [],
        "visited_urls": [],
        "fetched_pages": [],
        "iteration": 0,
        "phase": "collect",
        "profile": None,
        "user_decision": None,
        "selected_candidate_index": None,
        "guard_block": None,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_bridge.py -v`
Expected: PASS (existing cases + the new one).

- [ ] **Step 5: Commit**

```bash
git add app/graph/bridge.py tests/test_bridge.py
git commit -m "feat(history): fresh_search_input to restart a finished thread in place"
```

---

### Task 7: Wire `app/main.py` — data layer, thread unification, resume, follow-up, auto-tag

**Files:**
- Modify: `app/main.py` (imports, data-layer registration, `_ensure_graph`, `_thread_config`, `on_chat_resume`, `on_message` else-branch, `_persist_if_done`)
- Test: `tests/test_main_wiring.py`

**Interfaces:**
- Consumes: `init_chat_history_db`, `build_data_layer` (Task 2/3); `compute_auto_tags`, `render_tag_line` (Task 4); `derive_session_state` (Task 5); `fresh_search_input` (Task 6); `chainlit.data.get_data_layer`; `chainlit.types.ThreadDict`.
- Produces: a Chainlit app whose `config.code.data_layer` and `config.code.on_chat_resume` are registered; graph `thread_id == cl.context.session.thread_id`.

- [ ] **Step 1: Update imports**

In `app/main.py`, remove the now-unused `import uuid` line (top of file).

Change the existing data/bridge imports block:

```python
from app.db.connection import init_db
from app.db.profiles import save_profile
from app.db.users import set_user_locale
from app.graph.bridge import (
    PendingInput,
    ResumeAnswer,
    build_resume_command,
    parse_identity_text,
    read_pending_input,
)
```

to:

```python
import chainlit.data as cl_data
from chainlit.types import ThreadDict

from app.db.chat_history import build_data_layer, init_chat_history_db
from app.db.connection import init_db
from app.db.profiles import save_profile
from app.db.users import set_user_locale
from app.graph.bridge import (
    PendingInput,
    ResumeAnswer,
    build_resume_command,
    fresh_search_input,
    parse_identity_text,
    read_pending_input,
)
from app.ui.session_state import derive_session_state
from app.ui.tags import compute_auto_tags, render_tag_line
```

- [ ] **Step 2: Register the data layer**

In `app/main.py`, directly below the `log = structlog.get_logger()` line, add:

```python
@cl_data.data_layer
def _build_chat_history_layer():
    """Register Chainlit's persistent chat-history store (per-user threads)."""
    return build_data_layer()
```

- [ ] **Step 3: Initialize the chat-history DB at startup**

In `_ensure_graph`, change:

```python
    settings = get_settings()
    await init_db()
```

to:

```python
    settings = get_settings()
    await init_db()
    await init_chat_history_db()
```

- [ ] **Step 4: Unify the thread id**

Replace the whole `_thread_config` function:

```python
def _thread_config() -> dict[str, Any]:
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        thread_id = str(uuid.uuid4())
        cl.user_session.set("thread_id", thread_id)
    return {"configurable": {"thread_id": thread_id}}
```

with:

```python
def _thread_config() -> dict[str, Any]:
    # The graph thread id IS Chainlit's persisted thread id, so resuming a
    # thread restores both the UI messages and the LangGraph checkpoint.
    return {"configurable": {"thread_id": cl.context.session.thread_id}}
```

- [ ] **Step 5: Add the resume callback**

In `app/main.py`, add directly after the `on_chat_start` function:

```python
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """Restore server-side session flags when a past thread is reopened.

    Chainlit replays the stored messages itself; we only need to recover
    `awaiting`/locale. The graph checkpoint is the source of truth for what the
    conversation is waiting for, so we derive it from the snapshot.
    """
    graph = await _ensure_graph()
    config = _thread_config()
    locale = _user_locale()
    snapshot = await graph.aget_state(config)
    pending = read_pending_input(snapshot, locale)
    sess = derive_session_state(pending)
    cl.user_session.set("awaiting", sess.awaiting)
    if sess.awaiting == "narrowing":
        cl.user_session.set("narrowing_attribute", sess.narrowing_attribute)
        cl.user_session.set("narrowing_candidate_count", sess.narrowing_candidate_count)
```

- [ ] **Step 6: Restart a finished thread in place (drop the uuid mint)**

In `on_message`, replace the final `else` branch:

```python
    else:
        # No active expectation — treat as a fresh start.
        cl.user_session.set("thread_id", str(uuid.uuid4()))
        parsed = parse_identity_text(text)
        graph_input = PeopleSearchState(query=cast(IdentityQuery, parsed), locale=_user_locale())
```

with:

```python
    else:
        # No active expectation (e.g. a finished thread) — start a new search in
        # the SAME thread, resetting transient state but keeping the history.
        parsed = parse_identity_text(text)
        graph_input = cast(PeopleSearchState, fresh_search_input(parsed, _user_locale()))
```

- [ ] **Step 7: Auto-tag on completion + read thread id from the session**

In `_persist_if_done`, change:

```python
    thread_id = cl.user_session.get("thread_id") or ""
```

to:

```python
    thread_id = cl.context.session.thread_id
```

Then change the success block:

```python
        await save_profile(
            user_id=user_id,
            thread_id=str(thread_id),
            full_name=profile.full_name,
            profile=profile.model_dump(mode="json"),
            sources=sources,
        )
        await cl.Message(content=t("profile_saved", _user_locale())).send()
    except Exception as exc:
        log.warning("persist_profile_failed", error=str(exc))
```

to:

```python
        await save_profile(
            user_id=user_id,
            thread_id=str(thread_id),
            full_name=profile.full_name,
            profile=profile.model_dump(mode="json"),
            sources=sources,
        )
        await cl.Message(content=t("profile_saved", _user_locale())).send()
    except Exception as exc:
        log.warning("persist_profile_failed", error=str(exc))
        return

    # Auto-tags: discoverable via the built-in search (echo line) and stored in
    # thread metadata for future faceting. NOTE: we use metadata, not the
    # data layer's `tags=` param, because SQLite cannot bind a Python list.
    locale = _user_locale()
    tags = compute_auto_tags(profile.model_dump(mode="json"), locale)
    data_layer = cl_data.get_data_layer()
    if data_layer is not None:
        try:
            await data_layer.update_thread(str(thread_id), metadata={"auto_tags": tags})
        except Exception as exc:
            log.warning("thread_tags_failed", error=str(exc))
    await cl.Message(content=render_tag_line(tags, locale)).send()
```

- [ ] **Step 8: Write the wiring smoke test**

Create `tests/test_main_wiring.py`:

```python
"""Smoke test: importing app.main registers the data layer + resume callback."""

from __future__ import annotations


def test_main_registers_data_layer_and_resume() -> None:
    import app.main  # noqa: F401  (import triggers the chainlit decorators)
    from chainlit.config import config

    assert config.code.data_layer is not None
    assert config.code.on_chat_resume is not None
    # The factory returns the SQLAlchemy data layer over our SQLite file.
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    layer = config.code.data_layer()
    assert isinstance(layer, SQLAlchemyDataLayer)
    assert layer._conninfo.startswith("sqlite+aiosqlite:///")
```

- [ ] **Step 9: Run the wiring test**

Run: `uv run pytest tests/test_main_wiring.py -v`
Expected: PASS.

- [ ] **Step 10: Run the full quick suite + linters**

Run: `uv run pytest`
Expected: PASS (no regressions).

Run: `uv run ruff check . && uv run mypy app`
Expected: clean (no new errors). If ruff flags an unused import, ensure `uuid` was removed and `cast`/`PeopleSearchState`/`IdentityQuery` are still used.

- [ ] **Step 11: Commit**

```bash
git add app/main.py tests/test_main_wiring.py
git commit -m "feat(history): wire data layer, unified thread_id, resume, follow-up, auto-tags"
```

---

### Task 8: Docker persistence + docs

**Files:**
- Modify: `Dockerfile:78` (add `VOLUME`)
- Modify: `DEV.md` (new section)
- Modify: `README.md` (mention the new DB file)
- Test: manual / inspection

**Interfaces:** none (deployment + docs).

- [ ] **Step 1: Declare the data volume in the Dockerfile**

In `Dockerfile`, change:

```dockerfile
EXPOSE 8000
```

to:

```dockerfile
# Persist SQLite state (app.db, checkpoints, chat_history.db + WAL sidecars).
VOLUME ["/app/data"]

EXPOSE 8000
```

- [ ] **Step 2: Verify the schema file ships in the image build context**

Run: `git ls-files app/db/chat_history_schema.sql`
Expected: prints the path (git-tracked ⇒ copied by `COPY app ./app` and included in the hatchling wheel, exactly like the existing `app/db/schema.sql`). No pyproject change is needed.

- [ ] **Step 3: Document in `DEV.md`**

Add a new top-level section to `DEV.md` (after the existing content):

```markdown
## Chat history (persistence)

The Chainlit frontend persists per-user conversations via Chainlit's
`SQLAlchemyDataLayer`, stored in a **separate** SQLite file
`data/chat_history.db` (set by `CHAT_HISTORY_DB_PATH`). It is separate from
`data/app.db` because Chainlit's data-layer schema defines its own `users`
table, which would collide with the auth `users` table.

- Schema: `app/db/chat_history_schema.sql`, applied idempotently by
  `app/db/chat_history.py::init_chat_history_db()` at startup (WAL mode).
- The graph `thread_id` is unified with Chainlit's persisted thread id
  (`cl.context.session.thread_id`), so reopening a thread restores both the UI
  messages and the LangGraph checkpoint. `on_chat_resume` recovers the
  `awaiting` flag from the checkpoint.
- Reopening a finished search and typing a new name continues in the same
  thread via `fresh_search_input` (transient state reset, history preserved).
- **Search:** Chainlit's built-in sidebar search (substring over message text).
- **Auto-tags:** on completion, `platform:*`, `confidence:*`, `locale:*` tags
  are echoed as a searchable line and stored in thread metadata
  (`auto_tags`). They are NOT written via the data layer's `tags=` parameter —
  SQLite cannot bind a Python list, so that path is unused.
- **Docker:** `data/chat_history.db` (and its `-wal`/`-shm` sidecars) persist
  through the existing `./data:/app/data` mount in `docker-compose.yml`; the
  `Dockerfile` also declares `VOLUME ["/app/data"]`.
```

- [ ] **Step 4: Mention the new file in `README.md`**

In `README.md`, find the project-layout / storage description that mentions `data/app.db` and add a sibling bullet:

```markdown
- `data/chat_history.db` — persistent per-user Chainlit chat history (separate
  from `app.db`; see DEV.md → "Chat history").
```

(If no such list exists, add the line under the section that first introduces `data/app.db`.)

- [ ] **Step 5: Verify the edits**

Run: `grep -n 'VOLUME' Dockerfile && grep -n 'chat_history.db' DEV.md README.md`
Expected: the `VOLUME` line and the DEV.md/README mentions are present.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile DEV.md README.md
git commit -m "docs(history): document chat history + declare data VOLUME"
```

---

## Manual verification (after Task 7)

These cannot be unit-tested; run once before merging:

1. `uv run chainlit run app/main.py --port 8000`, log in as a seeded user.
2. Run a full search to completion → confirm the profile saves and a `🏷 Tags: …` line appears.
3. Reload the page → the conversation appears in the left **history sidebar**.
4. Click the past thread → messages are restored.
5. Type a new name in the restored (finished) thread → a new search starts in the same thread, old messages still above.
6. Mid-search (at a narrowing/confirm prompt), reload and reopen the thread → the next reply is still interpreted correctly (awaiting restored).
7. Type a platform (e.g. `linkedin`) in the sidebar search box → threads tagged with it are filtered in.

---

## Self-review notes

- **Spec coverage:** data layer (T2/3/7), separate DB (T1/2), thread unification (T7), resume (T5/7), follow-up-after-done (T6/7), built-in search (T8 docs; no code needed), auto-tags + searchable discovery (T4/7), Docker (T8), i18n (T4). All spec sections map to a task.
- **SQLite list-binding bug:** handled by storing tags in `metadata.auto_tags` (JSON string) instead of the `tags=` parameter.
- **Type consistency:** `fresh_search_input`, `compute_auto_tags`, `render_tag_line`, `derive_session_state`, `SessionState`, `init_chat_history_db`, `build_data_layer` names are used identically across tasks.
