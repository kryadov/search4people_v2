"""Graph state TypedDict used by LangGraph nodes."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Phase = Literal[
    "collect",
    "preliminary",
    "evaluate",
    "narrow",
    "expand",
    "fetch",
    "build",
    "confirm",
    "done",
]


class IdentityQuery(TypedDict, total=False):
    first_name: str
    last_name: str
    age: int
    country: str
    city: str
    school: str
    university: str
    employer: str
    extra: str  # free-form distinguishing detail


class PeopleSearchState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    query: IdentityQuery
    # Candidates are kept as plain dicts to stay msgpack-serializable for the
    # LangGraph checkpointer. Use `Candidate.model_validate(d)` when you need
    # the typed view inside a node.
    candidates: list[dict[str, Any]]
    visited_urls: list[str]
    fetched_pages: list[dict[str, Any]]
    iteration: int
    max_iterations: int
    phase: Phase
    locale: Literal["en", "ru"]
    # Stored as a dict for the same checkpointing reason; rendered via
    # `PersonProfile.model_validate(state["profile"])` in the UI layer.
    profile: dict[str, Any] | None
    user_decision: Literal["approve", "more", "switch_candidate", "abort"] | None
    selected_candidate_index: int | None
    # Set by a guardrail block; routes the graph straight to END.
    guard_block: dict[str, Any] | None
