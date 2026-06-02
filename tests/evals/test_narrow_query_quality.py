"""Eval: the narrowing planner asks a clear, discriminating question."""

from __future__ import annotations

import json
import pathlib

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.graph.nodes import plan_narrowing
from tests.evals.judge import LangChainJudge

DATA = pathlib.Path(__file__).parent / "data"

VALID_ATTRIBUTES = {
    "age", "country", "city", "school", "university",
    "employer", "profession", "distinctive_event",
}


@pytest.mark.eval
@pytest.mark.asyncio
async def test_narrowing_question_is_discriminating_en():
    candidates = json.loads(
        (DATA / "candidates" / "jane_doe.json").read_text(encoding="utf-8")
    )
    query = {"first_name": "Jane", "last_name": "Doe"}

    plan = await plan_narrowing(candidates, query, locale="en")

    # Deterministic structural checks:
    assert plan["attribute"] in VALID_ATTRIBUTES
    assert plan["question"].strip()

    test_case = LLMTestCase(
        input=(
            "Given several candidates named Jane Doe (GitHub/Globex, "
            "LinkedIn/Acme, Twitter), the assistant must ask ONE question to "
            "tell them apart."
        ),
        actual_output=f"attribute={plan['attribute']}; question={plan['question']}",
        retrieval_context=[json.dumps(candidates, ensure_ascii=False)],
    )
    quality = GEval(
        name="DiscriminatingQuestion",
        criteria=(
            "The question must be a single, clear, polite question in English "
            "that asks for an attribute genuinely useful to distinguish between "
            "the candidates (e.g. employer or city), not a generic or redundant "
            "one."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=LangChainJudge(),
        threshold=0.6,
        async_mode=False,
    )
    assert_test(test_case, [quality], run_async=False)
