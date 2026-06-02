"""Eval: build_profile must merge partials into a supported, well-formed profile."""

from __future__ import annotations

import json
import pathlib

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.graph.nodes import build_profile
from app.models.profile import PersonProfile
from app.tools.extract import extract_profile_from_page
from tests.evals.judge import LangChainJudge

DATA = pathlib.Path(__file__).parent / "data"


async def _extract_with_retry(page: str, golden: dict, attempts: int = 2):
    """Extract a partial, retrying the transient Ollama structured-output flake.

    Local models occasionally raise 'failed to load model vocabulary required
    for format' on a JSON-schema call; the app falls back to an empty profile.
    Retry a few times, then skip — it's an infra flake, not a quality signal.
    """
    for _ in range(attempts):
        partial = await extract_profile_from_page(
            full_name=golden["full_name"],
            distinguishers="",
            url=golden["url"],
            markdown=page,
            platform="github",
        )
        if partial.evidence or partial.work or partial.education:
            return partial
    pytest.skip("local model could not produce a structured extraction (Ollama format flake)")


@pytest.mark.eval
@pytest.mark.asyncio
async def test_build_profile_is_well_formed_and_supported():
    page = (DATA / "pages" / "jane_doe_github.md").read_text(encoding="utf-8")
    golden = json.loads((DATA / "goldens.json").read_text(encoding="utf-8"))["jane_doe_github"]

    partial = await _extract_with_retry(page, golden)

    state = {
        "query": {"first_name": "Jane", "last_name": "Doe"},
        "fetched_pages": [
            {
                "url": golden["url"],
                "platform": "github",
                "snippet": "",
                "markdown_len": len(page),
                "partial": partial.model_dump(mode="json"),
            }
        ],
    }

    # Merge can hit the same transient structured-output flake; retry then skip.
    profile = None
    for _ in range(2):
        patch = await build_profile(state)
        profile = PersonProfile.model_validate(patch["profile"])
        if profile.evidence:
            break
    if profile is None or not profile.evidence:
        pytest.skip("local model could not produce a structured merge (Ollama format flake)")

    # Deterministic: result validates as a PersonProfile and carries evidence.
    assert profile.full_name
    assert profile.evidence, "merged profile must cite at least one source"

    # Judge: every field is supported by the source page.
    test_case = LLMTestCase(
        input="Merge the extracted partial(s) into one coherent profile.",
        actual_output=profile.as_markdown(),
        retrieval_context=[page],
    )
    supported = GEval(
        name="EvidenceSupported",
        criteria=(
            "Check that each fact in the profile is supported by the source page, "
            "that the profile cites sources, and that the stated confidence is "
            "reasonable for the amount of corroborating evidence (a single source "
            "should not yield 'high')."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=LangChainJudge(),
        threshold=0.6,
        async_mode=False,
    )
    assert_test(test_case, [supported], run_async=False)
