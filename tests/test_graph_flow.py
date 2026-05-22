"""Integration test: drive the graph with stubbed LLM + stubbed search/fetch.

Confirms the high-level routing:
- single candidate path goes straight to fetch → build → confirm,
- the graph pauses on confirm_profile waiting for the user.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.build import build_graph
from app.models.profile import Evidence, PersonProfile
from app.models.state import PeopleSearchState
from app.tools.search import SearchHit


@pytest.mark.asyncio
async def test_single_candidate_flow_pauses_on_confirm(monkeypatch):
    # 1) Force preliminary_search to return exactly one candidate per platform call.
    async def fake_search_many_platforms(platforms, query, limit_per_platform=5):
        return [
            SearchHit(
                url="https://github.com/jdoe",
                title="John Doe",
                snippet="dev",
                platform="github",
            )
        ]

    monkeypatch.setattr("app.graph.nodes.search_many_platforms", fake_search_many_platforms)

    # 2) Stub the page fetcher so we never touch the network.
    class FakeFetchResult:
        def __init__(self):
            self.markdown = "John Doe is a developer."
            self.status = 200
            self.used_playwright = False
            self.blocked_by_robots = False
            self.url = "https://github.com/jdoe"
            self.error = None

    class FakeFetcher:
        async def fetch(self, url):
            return FakeFetchResult()

        async def close(self):
            return None

    monkeypatch.setattr("app.graph.nodes._get_fetcher", lambda: FakeFetcher())

    # 3) Stub extract_profile_from_page so we don't call an LLM.
    async def fake_extract(**kwargs):
        return PersonProfile(
            full_name=kwargs["full_name"],
            evidence=[Evidence(url=kwargs["url"], platform=kwargs["platform"], snippet="ok")],
            confidence="medium",
        )

    monkeypatch.setattr("app.graph.nodes.extract_profile_from_page", fake_extract)

    # 4) Stub build_chat_model so build_profile doesn't try to hit Anthropic.
    fake_model = MagicMock()
    fake_model.with_structured_output = MagicMock(return_value=fake_model)
    fake_model.ainvoke = AsyncMock(
        return_value=PersonProfile(
            full_name="John Doe",
            evidence=[Evidence(url="https://github.com/jdoe", platform="github", snippet="ok")],
            confidence="medium",
        )
    )
    monkeypatch.setattr("app.graph.nodes.build_chat_model", lambda **kw: fake_model)

    graph = build_graph().compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}}
    initial: PeopleSearchState = {
        "query": {"first_name": "John", "last_name": "Doe"},
        "locale": "en",
    }

    visited_nodes: list[str] = []
    async for event in graph.astream(initial, config=config, stream_mode="updates"):
        for node in event.keys():
            visited_nodes.append(node)

    # We should have walked: collect_identity → preliminary_search → evaluate
    # → fetch_pages → build_profile → confirm_profile (and paused there).
    assert visited_nodes[0] == "collect_identity"
    assert "preliminary_search" in visited_nodes
    assert "evaluate_candidates" in visited_nodes
    assert "fetch_pages" in visited_nodes
    assert "build_profile" in visited_nodes
    # confirm_profile runs but pauses on interrupt — the snapshot shows it.
    snapshot = await graph.aget_state(config)
    assert any(t.interrupts for t in snapshot.tasks)
