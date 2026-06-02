"""Eval: extraction must stay faithful to the source page and not fabricate."""

from __future__ import annotations

import json
import pathlib

import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.tools.extract import extract_profile_from_page
from tests.evals.judge import LangChainJudge

DATA = pathlib.Path(__file__).parent / "data"


def _goldens() -> dict:
    return json.loads((DATA / "goldens.json").read_text(encoding="utf-8"))


@pytest.mark.eval
@pytest.mark.asyncio
async def test_extract_is_faithful_to_page():
    page = (DATA / "pages" / "jane_doe_github.md").read_text(encoding="utf-8")
    golden = _goldens()["jane_doe_github"]

    profile = await extract_profile_from_page(
        full_name=golden["full_name"],
        distinguishers="",
        url=golden["url"],
        markdown=page,
        platform="github",
    )

    test_case = LLMTestCase(
        input=f"Extract a factual profile for {golden['full_name']} from the page.",
        actual_output=profile.as_markdown(),
        retrieval_context=[page],
        context=[page],
    )

    judge = LangChainJudge()
    no_fabrication = GEval(
        name="NoFabrication",
        criteria=(
            "Determine whether every concrete claim in the actual output "
            "(employers, schools, locations, links, dates) is directly supported "
            "by the page in retrieval context. Penalize any invented fact."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        threshold=0.6,
        async_mode=False,
    )

    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=0.6, model=judge, async_mode=False),
            no_fabrication,
        ],
        run_async=False,
    )


@pytest.mark.eval
@pytest.mark.asyncio
async def test_extract_rejects_unrelated_page():
    """A page that is not about the target → empty profile, low confidence."""
    page = (DATA / "pages" / "not_jane_doe.md").read_text(encoding="utf-8")
    golden = _goldens()["not_jane_doe"]

    profile = await extract_profile_from_page(
        full_name=golden["full_name"],
        distinguishers="",
        url=golden["url"],
        markdown=page,
        platform="web",
    )

    # Deterministic structural assertions (no judge needed):
    assert profile.full_name  # name echoed back
    assert profile.confidence == "low"
    assert profile.education == []
    assert profile.work == []
