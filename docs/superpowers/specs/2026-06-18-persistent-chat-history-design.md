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

## Non-goals (YAGNI)

- Postgres / cloud (LiteralAI) data layers — SQLite only.
- Full-text search over history, tagging beyond Chainlit defaults.
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

### i18n
Thread titles auto-derive from the first user message (the person's name) — no
new strings required. Optionally override a thread's title with the resolved
`full_name` once a profile completes (nice-to-have, can be deferred). Any new
user-facing string added during implementation goes through `t()` with both
`en` and `ru` entries.

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
- The deterministic graph suite (`tests/test_graph_flow.py`, evals) is untouched.
- `uv run ruff check .` and `uv run mypy app` stay clean.

## Open implementation details (settled in the plan, not blockers)

- Exact mechanism for resetting graph state on a post-`done` follow-up
  (fresh input vs. `aupdate_state`).
- Whether to override thread titles with the resolved name on completion.
- The precise SQLite DDL column types matching Chainlit's `SQLAlchemyDataLayer`
  queries (verify against `sql_alchemy.py`'s SQL before finalizing).
