"""Deterministic unit test for the narrowing planner (no real LLM)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph import nodes


@pytest.mark.asyncio
async def test_plan_narrowing_parses_llm_json(monkeypatch):
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(
        return_value=MagicMock(
            content=(
                '{"attribute": "employer", '
                '"question_en": "Where do they work?", '
                '"question_ru": "Где они работают?", '
                '"options": ["Acme", "Acme", "Globex", "  "]}'
            )
        )
    )
    monkeypatch.setattr(nodes, "build_chat_model", lambda **kw: fake_model)

    plan = await nodes.plan_narrowing(
        candidates=[{"url": "https://x/1", "title": "t", "snippet": "s", "platform": "web"}],
        query={"first_name": "John", "last_name": "Doe"},
        locale="en",
    )

    assert plan["attribute"] == "employer"
    assert plan["question"] == "Where do they work?"
    # Deduped, stripped, empty dropped:
    assert plan["options"] == ["Acme", "Globex"]


@pytest.mark.asyncio
async def test_plan_narrowing_falls_back_on_bad_json(monkeypatch):
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=MagicMock(content="not json at all"))
    monkeypatch.setattr(nodes, "build_chat_model", lambda **kw: fake_model)

    plan = await nodes.plan_narrowing(
        candidates=[],
        query={"first_name": "John", "last_name": "Doe"},
        locale="ru",
    )

    assert plan["attribute"] is None
    assert isinstance(plan["question"], str) and plan["question"]
    assert plan["options"] == []
