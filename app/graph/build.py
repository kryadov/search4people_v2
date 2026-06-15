"""Assemble and compile the LangGraph StateGraph."""

from __future__ import annotations

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    build_profile,
    collect_identity,
    confirm_profile,
    evaluate_candidates,
    expand_search,
    fetch_pages,
    narrow_query,
    preliminary_search,
    route_after_collect,
    route_after_confirm,
    route_after_evaluate,
    route_after_narrow,
)
from app.models.state import PeopleSearchState


def build_graph() -> StateGraph:
    """Wire the people-search graph (uncompiled)."""
    graph = StateGraph(PeopleSearchState)

    graph.add_node("collect_identity", collect_identity)
    graph.add_node("preliminary_search", preliminary_search)
    graph.add_node("evaluate_candidates", evaluate_candidates)
    graph.add_node("narrow_query", narrow_query)
    graph.add_node("expand_search", expand_search)
    graph.add_node("fetch_pages", fetch_pages)
    graph.add_node("build_profile", build_profile)
    graph.add_node("confirm_profile", confirm_profile)

    graph.add_edge(START, "collect_identity")
    graph.add_conditional_edges(
        "collect_identity",
        route_after_collect,
        {"preliminary_search": "preliminary_search", "blocked": END},
    )
    graph.add_edge("preliminary_search", "evaluate_candidates")

    graph.add_conditional_edges(
        "evaluate_candidates",
        route_after_evaluate,
        {
            "narrow_query": "narrow_query",
            "expand_search": "expand_search",
            "fetch_pages": "fetch_pages",
        },
    )

    # After narrowing, either jump straight to fetch (if the user picked a
    # specific candidate) or re-run the preliminary search with refined attrs.
    graph.add_conditional_edges(
        "narrow_query",
        route_after_narrow,
        {
            "preliminary_search": "preliminary_search",
            "fetch_pages": "fetch_pages",
            "blocked": END,
        },
    )
    # After expanding, re-evaluate candidates.
    graph.add_edge("expand_search", "evaluate_candidates")

    graph.add_edge("fetch_pages", "build_profile")
    graph.add_edge("build_profile", "confirm_profile")

    graph.add_conditional_edges(
        "confirm_profile",
        route_after_confirm,
        {
            "narrow_query": "narrow_query",
            "fetch_pages": "fetch_pages",
            "__end__": END,
        },
    )

    return graph


def make_checkpointer(db_path: str) -> AsyncSqliteSaver:
    """Build an AsyncSqliteSaver bound to the shared SQLite file.

    Returned as a context manager — callers must use `async with`.
    """
    return AsyncSqliteSaver.from_conn_string(db_path)  # type: ignore[return-value]
