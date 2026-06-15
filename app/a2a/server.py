"""A2A server: Agent Card, ASGI app assembly, and the `s4p-a2a` entrypoint.

Runs as a standalone process reusing the compiled people-search graph and the
shared SQLite database. Authenticates via Bearer tokens and exposes a single
skill, `people_search`.
"""

from __future__ import annotations

import contextlib
from typing import Any

import structlog
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityScheme,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.a2a.auth import BearerAuthMiddleware, current_user_id
from app.a2a.executor import PeopleSearchExecutor
from app.a2a.task_store import SqliteTaskStore
from app.config import get_settings
from app.db.connection import init_db
from app.graph.build import build_graph

log = structlog.get_logger()

_PROJECT_VERSION = "0.1.0"


def build_agent_card() -> AgentCard:
    settings = get_settings()
    url = settings.a2a_public_url or f"http://{settings.a2a_host}:{settings.a2a_port}/"
    return AgentCard(
        name="search4people",
        description="Conversational OSINT-style people search. Public data only.",
        url=url,
        version=_PROJECT_VERSION,
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        security_schemes={
            "bearer": SecurityScheme(root=HTTPAuthSecurityScheme(scheme="bearer"))
        },
        security=[{"bearer": []}],
        skills=[
            AgentSkill(
                id="people_search",
                name="Find a person and build a profile",
                description=(
                    "Given a name (+ optional hints), searches public platforms, "
                    "asks clarifying questions when ambiguous, and returns a cited "
                    "PersonProfile. Public information only."
                ),
                tags=["osint", "people-search", "profile"],
                input_modes=["text/plain"],
                output_modes=["application/json"],
            )
        ],
    )


@contextlib.asynccontextmanager
async def _lifespan_graph():
    """Enter the AsyncSqliteSaver once and compile the graph against it."""
    settings = get_settings()
    await init_db()
    async with AsyncSqliteSaver.from_conn_string(str(settings.db_path)) as saver:
        graph = build_graph().compile(checkpointer=saver)
        yield graph


async def build_app() -> Any:
    """Build the A2A Starlette app. Holds the compiled graph for the app's life."""
    # Enter the checkpointer context and keep it open for the process lifetime.
    graph_ctx = _lifespan_graph()
    graph = await graph_ctx.__aenter__()

    executor = PeopleSearchExecutor(graph, current_user_id=current_user_id)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=SqliteTaskStore(),
    )
    a2a_app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=handler,
    )

    # Pass lifespan to Starlette (v1 API) via build() **kwargs so shutdown
    # properly releases the AsyncSqliteSaver connection.
    @contextlib.asynccontextmanager
    async def _lifespan(app: Any):
        yield
        with contextlib.suppress(Exception):
            await graph_ctx.__aexit__(None, None, None)

    starlette_app = a2a_app.build(lifespan=_lifespan)

    # Wrap with BearerAuthMiddleware as a pure-ASGI wrapper (instead of
    # add_middleware) to avoid Starlette's "cannot add middleware after startup"
    # restriction and to guarantee the contextvar is set before the handler runs.
    return BearerAuthMiddleware(starlette_app)


def main() -> None:
    settings = get_settings()
    log.info("a2a_server_starting", host=settings.a2a_host, port=settings.a2a_port)

    import asyncio

    async def _serve() -> None:
        app = await build_app()
        config = uvicorn.Config(app, host=settings.a2a_host, port=settings.a2a_port)
        await uvicorn.Server(config).serve()

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
