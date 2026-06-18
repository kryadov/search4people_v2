"""extract_profile_from_page must surface the search title+snippet to the LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.profile import PersonProfile
from app.tools import extract as extract_mod


@pytest.mark.asyncio
async def test_extract_includes_search_result_block(monkeypatch):
    captured: dict = {}

    async def fake_ainvoke(messages):
        captured["messages"] = messages
        return PersonProfile(full_name="Konstantin Riadov", confidence="low")

    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    monkeypatch.setattr(extract_mod, "build_structured_model", lambda *a, **k: fake_model)

    await extract_mod.extract_profile_from_page(
        full_name="Konstantin Riadov",
        distinguishers="",
        url="https://www.linkedin.com/in/konstantin-riadov",
        markdown="",  # page body blocked
        platform="linkedin",
        title="Konstantin Riadov - Senior Engineer at Acme - Berlin",
        snippet="Senior Engineer at Acme. Based in Berlin. Previously at Foo.",
    )

    human = captured["messages"][1].content
    assert "SEARCH RESULT" in human
    assert "Senior Engineer at Acme" in human
    assert "Konstantin Riadov - Senior Engineer at Acme - Berlin" in human


@pytest.mark.asyncio
async def test_extract_without_snippet_still_works(monkeypatch):
    captured: dict = {}

    async def fake_ainvoke(messages):
        captured["messages"] = messages
        return PersonProfile(full_name="Jane Doe", confidence="low")

    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(side_effect=fake_ainvoke)
    monkeypatch.setattr(extract_mod, "build_structured_model", lambda *a, **k: fake_model)

    result = await extract_mod.extract_profile_from_page(
        full_name="Jane Doe",
        distinguishers="",
        url="https://github.com/jane-doe",
        markdown="# Jane Doe\nSoftware engineer.",
        platform="github",
    )

    assert isinstance(result, PersonProfile)
    human = captured["messages"][1].content
    assert "SEARCH RESULT" in human  # block present even when empty
    assert "(none)" in human


@pytest.mark.asyncio
async def test_extract_returns_low_confidence_when_model_returns_none(monkeypatch):
    # Structured-output models can return None when they produce no tool call
    # (e.g. a blocked page with an empty body). Must not raise.
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=None)
    monkeypatch.setattr(extract_mod, "build_structured_model", lambda *a, **k: fake_model)

    result = await extract_mod.extract_profile_from_page(
        full_name="Konstantin Riadov",
        distinguishers="",
        url="https://www.linkedin.com/in/konstantin-riadov",
        markdown="",
        platform="linkedin",
        snippet="Senior Engineer at Acme.",
    )

    assert isinstance(result, PersonProfile)
    assert result.full_name == "Konstantin Riadov"
    assert result.confidence == "low"
