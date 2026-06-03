"""build_profile must cite sources deterministically.

The merge LLM sometimes drops the `evidence` array; provenance is already known
from the input partials, so build_profile backfills it in code (no real LLM).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph import nodes
from app.models.profile import Evidence, PersonProfile


@pytest.mark.asyncio
async def test_build_profile_backfills_evidence_from_partials(monkeypatch):
    # Merge model returns a valid profile but WITHOUT evidence (LLM dropped it).
    merged_without_evidence = PersonProfile(full_name="Jane Doe", confidence="medium")
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=merged_without_evidence)
    monkeypatch.setattr(nodes, "build_structured_model", lambda *a, **k: fake_model)

    partial = PersonProfile(
        full_name="Jane Doe",
        evidence=[
            Evidence(url="https://github.com/jane-doe", platform="github", snippet="src")
        ],
        confidence="medium",
    )
    state = {
        "query": {"first_name": "Jane", "last_name": "Doe"},
        "fetched_pages": [
            {
                "url": "https://github.com/jane-doe",
                "platform": "github",
                "snippet": "",
                "markdown_len": 100,
                "partial": partial.model_dump(mode="json"),
            }
        ],
    }

    patch = await nodes.build_profile(state)
    profile = PersonProfile.model_validate(patch["profile"])

    urls = {str(ev.url).rstrip("/") for ev in profile.evidence}
    assert "https://github.com/jane-doe" in urls


@pytest.mark.asyncio
async def test_build_profile_dedupes_evidence(monkeypatch):
    # Merge model already kept the same source; backfill must not duplicate it.
    kept = PersonProfile(
        full_name="Jane Doe",
        evidence=[Evidence(url="https://github.com/jane-doe", platform="github")],
        confidence="medium",
    )
    fake_model = MagicMock()
    fake_model.ainvoke = AsyncMock(return_value=kept)
    monkeypatch.setattr(nodes, "build_structured_model", lambda *a, **k: fake_model)

    partial = PersonProfile(
        full_name="Jane Doe",
        evidence=[Evidence(url="https://github.com/jane-doe", platform="github")],
        confidence="medium",
    )
    state = {
        "query": {"first_name": "Jane", "last_name": "Doe"},
        "fetched_pages": [
            {
                "url": "https://github.com/jane-doe",
                "platform": "github",
                "snippet": "",
                "markdown_len": 100,
                "partial": partial.model_dump(mode="json"),
            }
        ],
    }

    patch = await nodes.build_profile(state)
    profile = PersonProfile.model_validate(patch["profile"])

    github_urls = [ev for ev in profile.evidence if "github.com/jane-doe" in str(ev.url)]
    assert len(github_urls) == 1
