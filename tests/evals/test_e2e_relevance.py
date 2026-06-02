"""Live e2e smoke: the graph should produce a profile about the intended person.

Opt-in: requires `-m "eval and live"`, network access, and a running judge.
Best-effort — tolerant threshold, single case.
"""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph.build import build_graph
from app.models.profile import PersonProfile
from tests.evals.judge import LangChainJudge

TARGET = {"first_name": "Guido", "last_name": "van Rossum"}


async def _run_to_completion(graph, config, initial):
    """Drive the graph, auto-approving any interrupt until it ends."""
    await graph.ainvoke(initial, config=config)
    # Resume through interrupts (identity already provided → narrow/confirm).
    for _ in range(6):
        snapshot = await graph.aget_state(config)
        if not any(t.interrupts for t in snapshot.tasks):
            break
        # Approve / accept defaults at every pause.
        await graph.ainvoke(Command(resume={"decision": "approve"}), config=config)
    return await graph.aget_state(config)


@pytest.mark.eval
@pytest.mark.live
@pytest.mark.asyncio
async def test_e2e_profile_is_about_target_person():
    graph = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "e2e-1"}}
    initial = {"query": TARGET, "locale": "en"}

    snapshot = await _run_to_completion(graph, config, initial)
    profile_dict = snapshot.values.get("profile")
    if not profile_dict:
        pytest.skip("graph did not yield a profile (live search returned nothing)")

    profile = PersonProfile.model_validate(profile_dict)
    test_case = LLMTestCase(
        input="Find a public profile for Guido van Rossum, creator of Python.",
        actual_output=profile.as_markdown(),
    )
    on_target = GEval(
        name="AboutTargetPerson",
        criteria=(
            "Judge whether the profile plausibly refers to Guido van Rossum, the "
            "creator of the Python programming language. It need not be complete, "
            "but it must not describe a clearly different person."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=LangChainJudge(),
        threshold=0.5,
        async_mode=False,
    )
    assert_test(test_case, [on_target], run_async=False)
