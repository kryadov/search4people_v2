# A2A Server Support — Design

**Date:** 2026-06-15
**Status:** Approved (brainstorming) → ready for implementation plan

## Goal

Expose search4people as an **A2A (Agent-to-Agent) server** so that other agents can
invoke the people-search capability as a remote skill. The existing Chainlit UI is
unchanged in behavior. Clarifying questions (currently surfaced to a human via
Chainlit) are surfaced to a calling agent via the A2A `input-required` task state.

## Key Decisions

| Decision | Choice |
|---|---|
| Direction | **Server only** (search4people answers A2A calls; not a client) |
| Deployment | **Separate process on its own port**, reusing the compiled graph + `data/app.db` |
| Protocol implementation | **Official `a2a-sdk`** (Python) — we write only the `AgentExecutor` |
| Auth | **Bearer token**, declared in the Agent Card `securitySchemes` |
| User binding | **Per-user API tokens** — a new `api_tokens` table maps `token → user_id` |
| Streaming | Support both `message/send` and `message/stream` (SSE) |
| Task identity | A2A `taskId` **is** the LangGraph `thread_id` (checkpointer handles resume) |

## Why `input-required` maps cleanly onto LangGraph `interrupt()`

`interrupt(payload)` freezes the graph, checkpoints state under `thread_id`, and waits
for `Command(resume=...)`. That is exactly the semantics of the A2A `input-required`
state: the agent pauses, returns a question, and resumes when the caller sends another
message on the same task.

| A2A | search4people (current) |
|---|---|
| `contextId` / `taskId` | `thread_id` (checkpointer key) |
| first `message/send` | start graph: `PeopleSearchState(query=...)` |
| `status.state = working` | `graph.astream(...)` running nodes |
| `status.state = input-required` + `status.message` | graph hit `interrupt()`; payload → question text |
| subsequent `message/send` (same task) | `Command(resume=...)` |
| `status.state = completed` + `artifact` | `phase == "done"`, profile built → artifact = `PersonProfile` |
| `status.state = failed` | exception / `not_found` |
| `tasks/cancel` | `abort` decision |

The three existing interrupts all become the same `input-required → answer → resume` loop:

| `PendingInput.kind` | A2A task state | parts emitted | resume payload |
|---|---|---|---|
| `ask_identity` | `input-required` | Text (question) | `{first_name, …}` |
| `ask_narrowing` | `input-required` | Text (question) + **Data** (candidates, options, attribute) | `{pick_index}` \| `{attribute, value}` \| `{extra}` |
| `confirm_profile` | `input-required` | Text + **Data** (profile) | `{decision: approve\|more\|abort, extra?}` |
| — (`phase=done`) | `completed` | **Artifact** = `PersonProfile` (Data) | — |

## Architecture

New **separate ASGI process** (`uv run s4p-a2a --port 8001`) reusing the same
`build_graph().compile(checkpointer=...)` and the same `data/app.db`. No dependency on
`chainlit.*`.

```
A2A client (another agent)
   │  JSON-RPC / SSE  + Bearer
   ▼
app/a2a/server.py  (a2a-sdk ASGI app, port 8001)
   ├─ AgentCard            → served at /.well-known/agent-card.json
   ├─ auth middleware      → Bearer → user_id (api_tokens)
   ├─ SqliteTaskStore      → thin, taskId ↔ thread_id
   └─ PeopleSearchExecutor (AgentExecutor)
          │  start / Command(resume)
          ▼
   app/graph (same compiled graph)  ──►  data/app.db (checkpoints + profiles + api_tokens)
```

**Isolation principle:** the protocol-neutral `interrupt ↔ resume` logic is extracted
out of `app/main.py` into a headless module `app/graph/bridge.py`, reused by **both**
Chainlit and the A2A server. Rendering (Markdown vs A2A parts) stays per-frontend.

### Modules

New / touched:

- `app/a2a/server.py` — build ASGI app, Agent Card, routes.
- `app/a2a/executor.py` — `PeopleSearchExecutor(AgentExecutor)`.
- `app/a2a/auth.py` — Bearer → `user_id` middleware.
- `app/a2a/task_store.py` — SQLite-backed `TaskStore` (taskId ↔ thread_id, survives restart).
- `app/graph/bridge.py` — headless core: interrupt → structured payload, answer → `Command(resume=...)`.
- `app/db/tokens.py` — token create/resolve/revoke.
- `app/db/schema.sql` — add `api_tokens` table.
- `app/scripts/create_token.py` — `uv run s4p-create-token <user> [--label]`.
- `app/config.py` — `a2a_host`, `a2a_port`, optional `a2a_public_url`.
- `app/main.py` — **refactor** to consume `bridge.py` (behavior-preserving).
- `pyproject.toml` — add `a2a-sdk` dependency; register `s4p-a2a`, `s4p-create-token` scripts.

## Component: `app/graph/bridge.py` (headless)

Extracts the protocol-neutral part of what is currently inline in `main.py`. Rendering
stays in the frontends; how a *human* phrases an answer (`#N`, `yes`, `да`) stays in
Chainlit; how an *agent* phrases an answer (structured `DataPart`) stays in the executor.

```python
@dataclass
class PendingInput:
    kind: Literal["ask_identity", "ask_narrowing", "confirm_profile"]
    question: str                 # human-readable (via i18n)
    data: dict[str, Any]          # candidates / options / attribute / profile
    candidate_count: int
    attribute: str | None

def read_pending_input(snapshot, locale) -> PendingInput | None:
    """Headless version of _handle_interrupt_and_render: pull the active interrupt
    out of the graph snapshot and normalize it. No cl.* references."""

def build_resume_command(pending: PendingInput, answer: ResumeAnswer) -> Command:
    """Headless version of the on_message branching: answer → Command(resume=...).
    Same branches as today: pick_index / {attribute, value} / decision."""
```

`ResumeAnswer` is a small union covering the three branches:
- identity → `{first_name, last_name, …}`
- narrowing → `{pick_index}` | `{attribute, value}` | `{extra}`
- confirm → `{decision, extra?}`

**Chainlit after refactor:** `main.py` calls `read_pending_input` for rendering and
`build_resume_command` for resume; parsing `#N` / `yes` / `да` stays in Chainlit.

**A2A server:** `executor.py` calls the same two functions but parses
`pick_index` / `decision` from JSON `DataPart`, not from a string.

**Constraint:** the `main.py` refactor must be **behavior-preserving** — existing graph
and UI tests pass unchanged. A golden test asserts the bridge produces the same
`Command` as the current inline code.

## Component: `PeopleSearchExecutor`

`a2a-sdk` calls `execute(context, event_queue)` and `cancel(...)`. Thin driver over
`graph.astream`:

```
execute(ctx, queue):
    user_id   = ctx.call_context.state["user_id"]      # from auth middleware
    thread_id = ctx.task_id                             # taskId == thread_id
    config    = {"configurable": {"thread_id": thread_id}}
    locale    = ctx.message.metadata.get("locale", "en")

    snapshot = await graph.aget_state(config)
    if snapshot has active interrupt:
        pending = read_pending_input(snapshot, locale)
        answer  = parse_a2a_message(ctx.message, pending)   # Data/Text → ResumeAnswer
        gin     = build_resume_command(pending, answer)
    else:
        query   = parse_a2a_message(ctx.message, None)      # name from first message
        gin     = PeopleSearchState(query=query, locale=locale)

    async for event in graph.astream(gin, config, stream_mode="updates"):
        queue.enqueue(TaskStatusUpdateEvent(state="working", node=...))   # SSE

    snapshot = await graph.aget_state(config)
    if pending := read_pending_input(snapshot, locale):
        queue.enqueue(status="input-required", message=render_a2a_parts(pending))
    elif phase == "done" and decision != abort:
        await save_profile(user_id=user_id, …)               # bind to user
        queue.enqueue(status="completed", artifact=profile_as_datapart)
    else:
        queue.enqueue(status="failed" | "completed-empty")   # not_found
```

Decisions baked in:
- `taskId == thread_id` → checkpointer handles resume; `tasks/resubscribe` reconstructs from the graph.
- Both `message/send` and `message/stream` run the same `astream`; the difference is whether interim `working` updates are emitted over SSE.
- `cancel` → write `abort` to the graph (resume on confirm) or mark canceled if not on a confirm interrupt.
- `locale` from `message.metadata.locale`, default `en`.
- `save_profile` is called server-side on `approve` (Chainlit does this in `_persist_if_done`; logic shared, call site per-frontend).

## Agent Card

Served by `a2a-sdk` at `/.well-known/agent-card.json`:

```jsonc
{
  "name": "search4people",
  "description": "Conversational OSINT-style people search. Public data only.",
  "url": "https://<host>:8001/",
  "version": "<from pyproject>",
  "capabilities": { "streaming": true, "pushNotifications": false },
  "defaultInputModes":  ["text/plain"],
  "defaultOutputModes": ["text/plain", "application/json"],
  "securitySchemes": { "bearer": { "type": "http", "scheme": "bearer" } },
  "security": [{ "bearer": [] }],
  "skills": [{
    "id": "people_search",
    "name": "Find a person and build a profile",
    "description": "Given a name (+ optional hints), searches public platforms, asks clarifying questions when ambiguous, and returns a cited PersonProfile.",
    "tags": ["osint", "people-search", "profile"],
    "inputModes":  ["text/plain"],
    "outputModes": ["application/json"]
  }]
}
```

One skill, `people_search`. The public-data ethics disclaimer rides along in the card `description`.

## Auth & tokens

**Middleware (`app/a2a/auth.py`):** reads `Authorization: Bearer <token>`, sha256-hashes
it, looks it up in `api_tokens`. Missing/invalid → JSON-RPC error `-32001` (HTTP 401 on
non-RPC paths). Valid → puts `user_id` into `call_context.state`. The agent card path
stays open (cards are public per spec).

**Schema (`app/db/schema.sql`):**

```sql
CREATE TABLE IF NOT EXISTS api_tokens (
    id          INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    token_hash  TEXT    NOT NULL UNIQUE,   -- sha256; plaintext shown once at creation
    label       TEXT,
    created_at  TEXT    NOT NULL,
    revoked_at  TEXT
);
```

`app/db/tokens.py`: `create_token(user_id, label) -> plaintext`,
`resolve_token(plaintext) -> user_id | None` (ignores rows with `revoked_at`),
`revoke_token(...)`.

**Hashing rationale:** token lookup must be deterministic (equality search), so sha256 —
not bcrypt (salted, can't be searched by equality). The token is a high-entropy random
string (`secrets.token_urlsafe(32)`), so sha256 is safe here — it is not a low-entropy
password.

**CLI (`app/scripts/create_token.py`):** `uv run s4p-create-token <username> [--label x]`,
prints the plaintext once. Registered in `pyproject.toml` `[project.scripts]` next to
`s4p-create-user`.

## Demo

`examples/a2a_demo.py`, run with `uv run python examples/a2a_demo.py` (assumes the server
is up on 8001 and a token is in `A2A_DEMO_TOKEN`). Uses the `a2a-sdk` client side and
walks the **full `input-required` cycle**, mirroring the Section "mapping" table:

```
1. Fetch Agent Card        → print skills / security
2. message/send "Jane Smith"
   → catches input-required (ask_narrowing), prints candidates from DataPart
3. message/send {pick_index: 1}   (same taskId)
   → input-required (confirm_profile), prints profile from DataPart
4. message/send {decision: "approve"}
   → completed, prints the PersonProfile artifact
```

Plus a short `message/stream` variant that prints interim `working` events (graph nodes
flowing through SSE).

## Testing

`tests/` in the existing style (network + LLM mocked, offline):

- `test_bridge.py` — `read_pending_input` / `build_resume_command` over all three
  interrupts; **golden test** that the bridge yields the same `Command` as the current
  inline `main.py` code (refactor safety net).
- `test_a2a_executor.py` — executor run with a stubbed graph: new task → `input-required`
  → resume → `completed`; profile artifact; `not_found` branch; `cancel` → abort.
- `test_a2a_auth.py` — valid / invalid / revoked token; agent-card openness; `user_id`
  propagation into `save_profile`.
- `test_tokens.py` — create / resolve / revoke, hash uniqueness.

## Documentation (update at the end)

- **README.md** — new A2A section (run the server, Agent Card, Bearer auth,
  `s4p-create-token`), `.env` vars (`A2A_*`), new entrypoint in the table, a **dedicated
  demo section** walking `examples/a2a_demo.py`, and a note on the `bridge.py` refactor in
  "Project layout" / "Extending".
- **`.env.example`** — `A2A_HOST`, `A2A_PORT` (+ describe `A2A_DEMO_TOKEN` for the demo;
  the Bearer token itself lives in the DB, not `.env`).

## Dependencies

`a2a-sdk` added to `pyproject.toml` as a main (not dev) dependency — the server is part
of the product.

## Out of scope (YAGNI)

- A2A **client** direction (calling other agents).
- Push notifications (`pushNotificationConfig`).
- OAuth2 / OIDC auth.
- Mounting A2A into the Chainlit ASGI app (kept as a separate process).
