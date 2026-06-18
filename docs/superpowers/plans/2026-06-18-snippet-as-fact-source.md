# Snippet-as-Fact-Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use a candidate's search `title` + `snippet` as a first-class fact source during extraction, so a selected candidate (notably LinkedIn) yields a populated profile even when the page body is blocked.

**Architecture:** Two focused edits. (1) `extract_profile_from_page` gains `title`/`snippet` params and shows them to the LLM in a `--- SEARCH RESULT ---` block. (2) `_fetch_and_extract` in the graph stops discarding a candidate whose page body is empty when a snippet/title exists, and always forwards `title`/`snippet` to extraction. No graph topology change.

**Tech Stack:** Python 3, LangChain/LangGraph, Pydantic, pytest (`pytest-asyncio`), `uv` for running.

## Global Constraints

- State values must be msgpack-serializable for the SQLite checkpointer — store dicts, not pydantic models (the partial is already `model_dump(mode="json")`).
- Follow existing patterns: nodes return a partial state patch (dict); LLM models are obtained via `build_structured_model(...)` and monkeypatched in tests.
- New extraction params are keyword-only with `None` defaults so existing callers/tests stay valid.
- Post-change checklist: `uv run pytest`, `uv run ruff check .`, `uv run mypy app` — no new errors.
- The graph topology is unchanged; do **not** edit `tests/test_graph_flow.py`.
- Spec: `docs/superpowers/specs/2026-06-18-snippet-as-fact-source-design.md`.

---

### Task 1: Extraction accepts and uses title + snippet

**Files:**
- Modify: `app/tools/extract.py`
- Test: `tests/test_extract_snippet.py` (create)

**Interfaces:**
- Consumes: `app.models.profile.PersonProfile`, `app.tools.extract.build_structured_model` (monkeypatch point).
- Produces: new signature
  `extract_profile_from_page(*, full_name: str, distinguishers: str, url: str, markdown: str, platform: str | None, title: str | None = None, snippet: str | None = None, max_chars: int = 12_000) -> PersonProfile`.
  The human message sent to the model contains a `--- SEARCH RESULT ---` block carrying `title`/`snippet` before the `--- PAGE MARKDOWN ---` block.

- [ ] **Step 1: Write the failing test**

Create `tests/test_extract_snippet.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_extract_snippet.py -v`
Expected: FAIL — `test_extract_includes_search_result_block` raises `TypeError: extract_profile_from_page() got an unexpected keyword argument 'title'`.

- [ ] **Step 3: Implement the new signature and message block**

In `app/tools/extract.py`, update the system prompt and the function. Replace the `_SYSTEM` constant body's middle paragraph and the function signature/message construction.

Update `_SYSTEM` to add a sentence about the search result (insert after the bullet list, before the "Return a partial..." paragraph):

```python
_SYSTEM = """You extract structured biographical facts about a single specific person from one web page.

You will receive:
- the person's first and last name (and possibly other distinguishing attributes),
- the search-result title and snippet for the page (a trustworthy summary), and
- a Markdown rendering of a web page that may or may not be about that person.

The search-result title and snippet are a legitimate source: when the page body
is empty or is a login / placeholder / "join to view" wall, rely on the title
and snippet. Do not fabricate beyond what the snippet and page support.

Return a partial PersonProfile containing ONLY fields that the page or the
search result actually supports. Leave fields you cannot back out of the text
empty or null. Always populate `evidence` with at least one entry citing the
page URL; when a fact comes from the snippet, set that evidence entry's
`snippet`. Do not fabricate.

If the page and the search result are clearly NOT about this person, return an
empty PersonProfile with `full_name` set to the queried name and
`confidence: "low"`.
"""
```

Update the function:

```python
async def extract_profile_from_page(
    *,
    full_name: str,
    distinguishers: str,
    url: str,
    markdown: str,
    platform: str | None,
    title: str | None = None,
    snippet: str | None = None,
    max_chars: int = 12_000,
) -> PersonProfile:
    """Run the LLM in structured-output mode against a single fetched page."""
    body = markdown[:max_chars]
    search_lines: list[str] = []
    if title:
        search_lines.append(f"Title: {title}")
    if snippet:
        search_lines.append(f"Snippet: {snippet}")
    search_block = "\n".join(search_lines) or "(none)"
    model = build_structured_model(PersonProfile, temperature=0.0)
    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(
            content=(
                f"Person: {full_name}\n"
                f"Distinguishing attributes: {distinguishers or '(none)'}\n"
                f"Source URL: {url}\n"
                f"Source platform: {platform or 'unknown'}\n\n"
                f"--- SEARCH RESULT ---\n{search_block}\n\n"
                f"--- PAGE MARKDOWN ---\n{body or '(page body unavailable)'}"
            )
        ),
    ]
    try:
        result = await model.ainvoke(messages)
    except Exception as exc:
        log.warning("extract_failed", url=url, error=str(exc))
        return PersonProfile(full_name=full_name, confidence="low")
    if isinstance(result, PersonProfile):
        return result
    # Some providers return a dict; coerce defensively.
    return PersonProfile.model_validate(result)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_extract_snippet.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint & type-check the touched file**

Run: `uv run ruff check app/tools/extract.py tests/test_extract_snippet.py && uv run mypy app`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add app/tools/extract.py tests/test_extract_snippet.py
git commit -m "feat(extract): use search title+snippet as a fact source"
```

---

### Task 2: fetch_pages keeps blocked candidates that have a snippet

**Files:**
- Modify: `app/graph/nodes.py` (`_fetch_and_extract` inside `fetch_pages`, ~lines 333-360)
- Test: `tests/test_fetch_pages_snippet.py` (create)

**Interfaces:**
- Consumes: Task 1's `extract_profile_from_page(..., title=..., snippet=...)`; `app.tools.fetch.FetchResult`; `nodes._get_fetcher`, `nodes.extract_profile_from_page`, `nodes.get_guardrails` (monkeypatch points).
- Produces: `fetch_pages` returns `fetched_pages` entries that now include `"fetch_blocked": bool`; a candidate with empty page body but a non-empty `snippet`/`title` is retained (not dropped). A candidate with neither body nor snippet/title is still dropped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fetch_pages_snippet.py`:

```python
"""fetch_pages must not discard a blocked candidate that carries a snippet."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph import nodes
from app.models.profile import Evidence, PersonProfile
from app.tools.fetch import FetchResult


def _noop_guardrails():
    g = MagicMock()
    # scan_content returns an object whose transformed_text is None (no change).
    g.scan_content = AsyncMock(return_value=MagicMock(transformed_text=None))
    return g


@pytest.mark.asyncio
async def test_blocked_candidate_with_snippet_is_kept(monkeypatch):
    # Fetcher returns an empty body (robots/auth-wall blocked).
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock(return_value=FetchResult(
        url="https://www.linkedin.com/in/konstantin-riadov",
        status=0,
        markdown="",
        used_playwright=False,
        blocked_by_robots=True,
    ))
    monkeypatch.setattr(nodes, "_get_fetcher", lambda: fetcher)
    monkeypatch.setattr(nodes, "get_guardrails", _noop_guardrails)

    captured: dict = {}

    async def fake_extract(**kwargs):
        captured.update(kwargs)
        return PersonProfile(
            full_name="Konstantin Riadov",
            bio="Senior Engineer at Acme, Berlin.",
            evidence=[Evidence(
                url="https://www.linkedin.com/in/konstantin-riadov",
                platform="linkedin",
                snippet="Senior Engineer at Acme. Based in Berlin.",
            )],
            confidence="low",
        )

    monkeypatch.setattr(nodes, "extract_profile_from_page", fake_extract)

    state = {
        "query": {"first_name": "Konstantin", "last_name": "Riadov"},
        "candidates": [{
            "url": "https://www.linkedin.com/in/konstantin-riadov",
            "title": "Konstantin Riadov - Senior Engineer at Acme - Berlin",
            "snippet": "Senior Engineer at Acme. Based in Berlin.",
            "platform": "linkedin",
        }],
        "selected_candidate_index": 0,
        "visited_urls": [],
        "fetched_pages": [],
    }

    patch = await nodes.fetch_pages(state)

    pages = patch["fetched_pages"]
    assert len(pages) == 1
    assert pages[0]["fetch_blocked"] is True
    assert pages[0]["markdown_len"] == 0
    # The snippet+title reached extraction.
    assert captured["snippet"] == "Senior Engineer at Acme. Based in Berlin."
    assert captured["title"] == "Konstantin Riadov - Senior Engineer at Acme - Berlin"


@pytest.mark.asyncio
async def test_blocked_candidate_without_snippet_is_dropped(monkeypatch):
    fetcher = MagicMock()
    fetcher.fetch = AsyncMock(return_value=FetchResult(
        url="https://example.com/nothing",
        status=0,
        markdown="",
        used_playwright=False,
    ))
    monkeypatch.setattr(nodes, "_get_fetcher", lambda: fetcher)
    monkeypatch.setattr(nodes, "get_guardrails", _noop_guardrails)

    async def fake_extract(**kwargs):  # must not be called
        raise AssertionError("extract should not run when there is nothing to extract")

    monkeypatch.setattr(nodes, "extract_profile_from_page", fake_extract)

    state = {
        "query": {"first_name": "Jane", "last_name": "Doe"},
        "candidates": [{
            "url": "https://example.com/nothing",
            "title": None,
            "snippet": None,
            "platform": None,
        }],
        "selected_candidate_index": 0,
        "visited_urls": [],
        "fetched_pages": [],
    }

    patch = await nodes.fetch_pages(state)
    assert patch["fetched_pages"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fetch_pages_snippet.py -v`
Expected: FAIL — `test_blocked_candidate_with_snippet_is_kept` fails because the current code does `if not result.markdown: return None`, so `fetched_pages` is empty (`IndexError`/`assert len == 1`).

- [ ] **Step 3: Rewrite `_fetch_and_extract`**

In `app/graph/nodes.py`, replace the body of the inner `_fetch_and_extract` (currently lines ~333-360) with:

```python
    async def _fetch_and_extract(c: dict[str, Any]) -> dict[str, Any] | None:
        url = c.get("url")
        if not url:
            return None
        title = c.get("title")
        snippet = c.get("snippet")
        async with sem:
            result = await fetcher.fetch(url)
            markdown = result.markdown or ""
            # Untrusted page content is a prompt-injection vector before it
            # reaches the extraction LLM; sanitize anything the guard flags.
            if markdown:
                scan = await get_guardrails().scan_content(markdown)
                if scan.transformed_text is not None:
                    markdown = scan.transformed_text
            # Bail only when there is nothing at all to extract from: a blocked
            # page (empty body) still yields facts from the search snippet/title.
            if not markdown and not (snippet or title):
                return None
            extracted = await extract_profile_from_page(
                full_name=full_name,
                distinguishers=distinguishers,
                url=url,
                markdown=markdown,
                platform=c.get("platform"),
                title=title,
                snippet=snippet,
            )
            return {
                "url": url,
                "platform": c.get("platform"),
                "snippet": snippet,
                "markdown_len": len(markdown),
                "fetch_blocked": not markdown,
                "partial": extracted.model_dump(mode="json"),
            }
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_fetch_pages_snippet.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full fast suite to confirm no regressions**

Run: `uv run pytest`
Expected: PASS (all green; heavy `eval`/`models` tests are deselected by `pyproject.toml` addopts).

- [ ] **Step 6: Lint & type-check**

Run: `uv run ruff check . && uv run mypy app`
Expected: no new errors.

- [ ] **Step 7: Commit**

```bash
git add app/graph/nodes.py tests/test_fetch_pages_snippet.py
git commit -m "fix(graph): keep blocked candidates that have a search snippet"
```

---

### Task 3 (optional): Eval golden for snippet-only extraction

Do this only if running the opt-in eval suite. It guards against fabrication when the only source is a snippet.

**Files:**
- Create: `tests/evals/data/pages/konstantin-riadov-snippet.md`
- Modify: `tests/evals/data/goldens.json`
- Modify: `tests/evals/test_extract_faithfulness.py` (add/parametrize a case)

**Interfaces:**
- Consumes: existing eval harness (`tests/evals/conftest.py` judge, `goldens.json` schema `full_name`/`url`/`expected_facts`/`forbidden_facts`).

- [ ] **Step 1: Add the snippet "page"**

Create `tests/evals/data/pages/konstantin-riadov-snippet.md` containing only the short search snippet text, e.g.:

```markdown
Konstantin Riadov - Senior Engineer at Acme - Berlin

Senior Engineer at Acme. Based in Berlin, Germany. Previously Software Engineer at Foo.
```

- [ ] **Step 2: Add the golden entry**

Add to `tests/evals/data/goldens.json` an object matching the existing schema:

```json
{
  "full_name": "Konstantin Riadov",
  "url": "https://www.linkedin.com/in/konstantin-riadov",
  "expected_facts": ["Senior Engineer", "Acme", "Berlin"],
  "forbidden_facts": ["CEO", "London"]
}
```

(Match the exact key names already used by neighboring entries in the file — read one first and mirror it.)

- [ ] **Step 3: Wire it into the faithfulness test**

In `tests/evals/test_extract_faithfulness.py`, add the new golden to the existing parametrization (mirror how an existing page is loaded and passed to `extract_profile_from_page`, supplying the snippet as `snippet=`). Reuse the existing `FaithfulnessMetric` + `GEval(NoFabrication)` assertion — do not invent a new metric.

- [ ] **Step 4: Run the eval (needs a judge model up)**

Run: `uv run pytest tests/evals/test_extract_faithfulness.py -m eval -v`
Expected: PASS, or SKIP if no judge is available (per `_require_judge`). Read any failing metric's `reason` before changing prompts.

- [ ] **Step 5: Commit**

```bash
git add tests/evals/data/pages/konstantin-riadov-snippet.md tests/evals/data/goldens.json tests/evals/test_extract_faithfulness.py
git commit -m "test(evals): golden for snippet-only extraction"
```

---

## Self-Review

**Spec coverage:**
- "Add `title`/`snippet` params + SEARCH RESULT block + system prompt change" → Task 1. ✓
- "`_fetch_and_extract` bail only when nothing to extract; forward title/snippet; `fetch_blocked` flag" → Task 2. ✓
- "Snippet used always, not only as fallback" → Task 1 always renders the block; Task 2 always forwards the values. ✓
- "Confidence stays honest (low/medium), fields populated" → no forcing logic added; covered by leaving LLM/merge to decide. ✓
- "Tests: extract message contains snippet; fetch_pages keeps blocked+snippet; regression drops empty+no-snippet; optional eval golden" → Tasks 1, 2, 3. ✓
- "Graph topology unchanged; don't touch test_graph_flow.py" → stated in Global Constraints. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. Task 3's golden JSON keys say "mirror an existing entry" because the file's exact schema must be read at edit time — the canonical keys (`full_name`, `url`, `expected_facts`, `forbidden_facts`) are taken verbatim from DEV.md.

**Type consistency:** `extract_profile_from_page(..., title=None, snippet=None)` defined in Task 1 is called with `title=`/`snippet=` in Task 2. `FetchResult` fields (`url`, `status`, `markdown`, `used_playwright`, `blocked_by_robots`) match `app/tools/fetch.py`. `fetched_pages` entry keys (`url`, `platform`, `snippet`, `markdown_len`, `fetch_blocked`, `partial`) are consistent across Task 2 code and test.
