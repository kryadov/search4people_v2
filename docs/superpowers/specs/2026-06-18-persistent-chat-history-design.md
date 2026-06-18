# Persistent per-user chat history — design

**Status:** approved (brainstorm), ready for implementation planning
**Date:** 2026-06-18
**Component:** Chainlit frontend (`app/main.py`), persistence (`app/db/`)

## Problem

The Chainlit frontend has no chat history. Each browser session mints a fresh
`thread_id` (`uuid4`) that lives only in `cl.user_session` (in-memory), so when
the session expires or the user reconnects, the conversation is gone: there is no
sidebar of past dialogs, no resume, no continuation.

Two facts about the current code make this tractable:

- The LangGraph graph is **already** checkpointed per `thread_id` via
  `AsyncSqliteSaver` in `data/app.db`. Graph state is durable; it's just keyed by
  a `thread_id` we throw away.
- `password_auth` already returns a stable `cl.User(identifier=username, …)`, so
  every request is tied to a known user.

What's missing is (a) durably tying threads to users with a browsable UI, and
(b) reusing the persisted `thread_id` so resume restores both the messages and
the graph state.

## Goal

Give each authenticated user the **full native Chainlit chat-history UX**: a
sidebar listing their past conversations, click-to-resume with all messages
restored and the conversation continuable, plus rename/delete. History is scoped
per user automatically by Chainlit's identifier.

## In scope (added after initial brainstorm)

- **Search over history** — satisfied by Chainlit's built-in sidebar search (see
  "Search and auto-tagging"). No custom search engine.
- **Auto-tagging** of threads with the searched person's platforms, the result
  confidence, and the locale, plus the ability to find threads by those tags.

## Non-goals (YAGNI)

- Postgres / cloud (LiteralAI) data layers — SQLite only.
- A custom full-text engine: **no SQLite FTS5 index, no search over structured
  profile fields.** Search relies on Chainlit's built-in substring search.
- **Manual / user-entered tags** — tagging is automatic only.
- Thread sharing / export.
- Changing the A2A frontend (it does not use Chainlit and is unaffected).
- Any change to graph topology, nodes, prompts, or guardrails.

## Approach

Register Chainlit's official `SQLAlchemyDataLayer` via `@cl.data_layer`. This
turns on Chainlit's built-in persistence: threads, steps (messages), elements,
and feedback are written to the DB, and the sidebar/resume UI lights up. Users
are matched by `identifier` (the username), which auth already provides.

The key integration: **Chainlit's persisted thread id becomes the graph
`thread_id`.** Resuming a thread then restores the UI messages (Chainlit) *and*
the graph's checkpoint state (LangGraph) with no extra bookkeeping.

### API confirmed against the installed version

- `chainlit>=2.11.1` (`.chainlit/config.toml` → `generated_by = "2.11.1"`).
- `chainlit.data.sql_alchemy.SQLAlchemyDataLayer(conninfo: str,
  ssl_require=False, storage_provider: Optional[BaseStorageClient] = None)` —
  `storage_provider=None` is valid (no binary element storage).
- `@cl.data_layer` registers the layer; `@cl.on_chat_resume(func)` receives a
  `ThreadDict`.
- `SQLAlchemyDataLayer` does **not** create its tables; the schema is our
  responsibility (no `.sql` ships with the package).
- `sqlalchemy` and `greenlet` are **not** currently installed — must be added.

## Storage layout

Chainlit's data-layer schema defines a table literally named `users` (UUID PK,
`identifier`, `metadata`, `createdAt`), which **collides** with the existing auth
`users` table. To avoid renaming the auth table (which would ripple through auth,
A2A tokens, and every accessor), chat-history persistence goes in a **separate
SQLite file**:

- **`data/app.db`** (unchanged): auth `users`, `profiles`, `source_evidence`,
  `api_tokens`, `a2a_tasks`, `guard_events`, **and the LangGraph checkpoint
  tables** (exactly as today).
- **`data/chat_history.db`** (new): the Chainlit data-layer tables — `users`,
  `threads`, `steps`, `elements`, `feedbacks`.

The two files share `thread_id` strings but are joined only logically (no foreign
key across files). Both run in **WAL mode** to tolerate concurrent writers (the
app's `aiosqlite` connections, the checkpointer, and the data-layer engine).

*Alternative considered:* one DB file by renaming the auth `users` table.
Rejected — invasive and touches unrelated subsystems.

### Schema management

A new SQLite DDL for the five Chainlit tables (Chainlit's documented schema,
SQLite-adapted: `TEXT` for UUIDs, `TEXT` for JSON/array columns, `INTEGER` for
booleans). Applied idempotently at startup, mirroring `app/db/connection.py`'s
`init_db()` pattern. Lives in a new module (e.g. `app/db/chat_history.py` with
`init_chat_history_db()` + a `chat_history_schema.sql`).

### Element storage

Profile cards are plain markdown messages — there are no binary file elements to
persist — so `storage_provider=None`. (Chainlit's spontaneous file-upload feature
is enabled in config but unused by this app's flow; if a user uploads a file it
simply won't be persisted to object storage, which is acceptable. Revisit only if
binary elements become part of the product.)

## Component changes

### `app/config.py`
Add:
```python
chat_history_db_path: Path = Path("data/chat_history.db")
```
On by default (consistent with the repo convention that heavy/standard features
ship enabled, not as optional extras). No enable/disable flag — keep it simple.

### `pyproject.toml`
Add to main `dependencies`:
```
"sqlalchemy>=2.0,<3.0",
"greenlet>=3.0",
```
(`aiosqlite` is already present.)

### New: `app/db/chat_history.py` + `app/db/chat_history_schema.sql`
- `init_chat_history_db()` — create the file, apply the DDL, set WAL.
- A small factory returning a configured `SQLAlchemyDataLayer` pointed at
  `sqlite+aiosqlite:///<chat_history_db_path>` with `storage_provider=None`.

### `app/main.py`

1. **Register the data layer.** Add a `@cl.data_layer` function that returns the
   `SQLAlchemyDataLayer` from the factory. Call `init_chat_history_db()` during
   startup (extend `_ensure_graph()` / the existing `init_db()` call site).

2. **Unify `thread_id`.** `_thread_config()` stops generating its own `uuid4`.
   It reads Chainlit's persisted thread id (`cl.context.session.thread_id`) and
   passes it straight into `{"configurable": {"thread_id": …}}`. If the id is not
   yet available at `on_chat_start`, it is by the first `on_message`, which is
   where the graph actually runs.

3. **`@cl.on_chat_resume(thread)`.** Chainlit auto-replays the persisted messages;
   this callback restores only **server-side session state** so the next reply
   routes correctly:
   - `locale` — from the authenticated user's metadata (already wired via
     `_user_locale()`).
   - `awaiting` (and, for the narrowing case, `narrowing_attribute` /
     `narrowing_candidate_count`) — **derived from the graph checkpoint**, which
     is the source of truth for "what is this conversation waiting for". Reuse the
     existing `aget_state` → `read_pending_input` logic (the same logic
     `_handle_interrupt_and_render` uses) to recompute `awaiting` without
     re-rendering the prompt. Factor that derivation into a small shared helper so
     both resume and the live interrupt path call it.

4. **Follow-up after a finished search ("continue in the same thread").**
   When a thread at `phase="done"` receives a new identity message, keep the same
   `thread_id` and start a fresh graph run, resetting the transient state keys
   (`query`, `candidates`, `profile`, `phase`, `guard_block`, `user_decision`,
   iteration counters, …) so the new search doesn't inherit stale data. The old
   result remains visible above (Chainlit messages are append-only); the new
   search appends below.

   The current `else` branch in `on_message` that mints a brand-new `uuid4`
   thread_id is **removed** — "start a new conversation" is now Chainlit's "New
   Chat" button, which creates a new thread id. The reset is implemented via the
   graph input / `aupdate_state` on the existing thread; the exact mechanism
   (fresh `PeopleSearchState` input vs. explicit state reset) is settled in the
   plan and pinned by a test.

5. **Auto-tag on completion.** In `_persist_if_done`, after the profile is saved,
   compute the tag set and write it to the thread (see "Search and auto-tagging").

### i18n
Thread titles auto-derive from the first user message (the person's name) — no
new strings required. Optionally override a thread's title with the resolved
`full_name` once a profile completes (nice-to-have, can be deferred). Any new
user-facing string added during implementation goes through `t()` with both
`en` and `ru` entries.

## Search and auto-tagging

### Search — built-in, no custom engine

`SQLAlchemyDataLayer.list_threads` already honors `ThreadFilter.search`: it
substring-matches the keyword (case-insensitive) against every step's `output`
across the user's threads, and the Chainlit sidebar exposes a search box wired to
it. Once the data layer is on, "search my past conversations by keyword" works
with **zero extra code**. Our only task is to confirm it functions against the
SQLite schema and that the searched person's names/platforms (already present in
the candidate list and profile-card messages) are therefore findable.

No FTS5, no profile-table search (explicit non-goals). The built-in scan is
O(threads × steps) in Python — fine for per-user history sizes; revisit only if
history grows large enough to matter.

### Auto-tagging

When a search completes (the `_persist_if_done` path, where we already hold the
final state + validated `PersonProfile`), compute a normalized tag set and write
it to the thread.

- **Tag computation** (namespaced tokens, so search is precise and collision-free):
  - `platform:<name>` for each distinct platform in `profile.evidence`
    (e.g. `platform:linkedin`, `platform:github`).
  - `confidence:<low|medium|high>` from the profile confidence.
  - `locale:<en|ru>` from the conversation locale.
- **Canonical storage:** persist to the thread's `tags` column via the data
  layer — `await get_data_layer().update_thread(thread_id, tags=[...])`. This is
  the source of truth and keeps a future facet-filter possible with no migration.
- **Discovery / "filter by tag":** `ThreadFilter` has no `tags` field and the
  sidebar has no tag-filter UI, so to make tags *findable* with the built-in
  search the user kept, also append a compact, searchable tag line to the thread
  on completion (a short `cl.Message`/step like
  `🏷 platform:linkedin · platform:github · confidence:high · locale:ru`). Typing
  `platform:linkedin` (or just `linkedin`) in the sidebar search box then surfaces
  the matching threads. Tags thus live in two places: canonically in
  `thread.tags`, and echoed once into searchable content for discovery.
  *(Decision point for spec review: this echo line is the pragmatic way to get
  tag filtering without forking the frontend; the alternative — a custom
  `/history <tag>` listing command — is heavier and duplicates the sidebar.)*

New i18n: the tag-line label (and any "tagged" copy) goes through `t()` in
`en`/`ru`. The tag tokens themselves stay untranslated (stable identifiers).

## Data flow

```
Login (password_auth → cl.User identifier=username)
  │
  ├─ on_chat_start: init graph + chat-history DB; greet; awaiting="identity"
  │     thread_id = cl.context.session.thread_id   (persisted by data layer)
  │
  ├─ on_message: graph runs on that thread_id
  │     LangGraph checkpoint → data/app.db
  │     Chainlit messages/steps → data/chat_history.db
  │     on completion: profile → app.db; auto-tags → thread.tags + searchable line
  │
  └─ Sidebar lists this user's threads (chat_history.db, scoped by identifier)
        │
        └─ click old thread → on_chat_resume(thread):
              Chainlit replays messages (chat_history.db)
              we restore awaiting/locale from the graph checkpoint (app.db)
              next reply continues the same thread_id
              (if phase=="done": next identity msg starts a fresh run, same thread)
```

## Error handling

- **DB lock contention:** WAL mode on both files; the data-layer engine and the
  app's connections are independent. The existing `_persist_if_done` already
  swallows persistence errors with a warning — keep that defensive posture for
  the new write paths.
- **Resume of a corrupt/absent checkpoint:** if `aget_state` returns nothing for
  a resumed thread (e.g. checkpoint lost but Chainlit thread present), fall back
  to `awaiting="identity"` so the user can start fresh in that thread rather than
  hitting a dead end.
- **Data layer init failure:** log and continue without persistence rather than
  crashing the server (history is an enhancement, not a hard dependency of a
  single search).

## Deployment (Docker)

The new `chat_history.db` lives under `data/`, which `docker-compose.yml` already
mounts (`./data:/app/data`), so **it persists across container restarts with no
compose change** — including the WAL sidecars (`chat_history.db-wal`, `-shm`).
Concrete to-dos:

- **Ship the schema file:** `app/db/chat_history_schema.sql` is copied by the
  existing `COPY app ./app` and read via `Path(__file__).with_name(...)` from the
  source tree (same as today's `schema.sql`). Also add `*.sql` to the hatchling
  wheel includes so the installed-package path keeps it too.
- **Make data persistence explicit:** add `VOLUME ["/app/data"]` to the
  `Dockerfile` so the directory persists even when run without compose, and so the
  intent is documented at the image level. (`init_db()` / `init_chat_history_db()`
  already `mkdir(parents=True, exist_ok=True)`, so the dir is created on first run.)
- No new ports, env, or system packages — `sqlalchemy`/`greenlet` are pure-Python
  wheels pulled in by `uv sync`.

## Testing

- **Unit — resume state restoration:** given a fake checkpoint snapshot in each
  pending state (`ask_identity`, `ask_narrowing`, `confirm_profile`, `done`,
  empty), the shared "derive awaiting from snapshot" helper returns the correct
  `awaiting` + narrowing fields. (`done`/empty → `awaiting="identity"`.)
- **Unit — follow-up reset:** feeding a new identity to a `done` thread produces a
  graph input/state with transient keys cleared and the same `thread_id`.
- **Smoke — lifecycle wiring:** `@cl.data_layer` returns a `SQLAlchemyDataLayer`
  pointed at the configured path; `init_chat_history_db()` is idempotent and
  creates the five tables.
- **Unit — auto-tag computation:** a sample final state + `PersonProfile`
  (mixed platforms / confidence / locale) yields the expected normalized,
  deduped, namespaced tag list, and the searchable tag-line renders those tokens.
- The deterministic graph suite (`tests/test_graph_flow.py`, evals) is untouched.
- `uv run ruff check .` and `uv run mypy app` stay clean.

## Open implementation details (settled in the plan, not blockers)

- Exact mechanism for resetting graph state on a post-`done` follow-up
  (fresh input vs. `aupdate_state`).
- Whether to override thread titles with the resolved name on completion.
- The precise SQLite DDL column types matching Chainlit's `SQLAlchemyDataLayer`
  queries (verify against `sql_alchemy.py`'s SQL before finalizing).
- Confirm the searchable tag-line approach vs. a custom `/history <tag>` command
  for tag discovery (spec leans to the tag-line; final call at review).
- How to obtain the data-layer handle inside `_persist_if_done`
  (`chainlit.data.get_data_layer()`), and timing so the thread row already exists
  when `update_thread(tags=…)` runs.
