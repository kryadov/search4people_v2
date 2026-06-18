# Use search title + snippet as a fact source in extraction

**Date:** 2026-06-18
**Status:** Approved (design)

## Problem

When a user selects a candidate whose page cannot be fetched (notably
LinkedIn), the final report comes back empty with `confidence: "low"` and every
field blank — even though the search result that was shown to the user contained
real facts (e.g. *"Konstantin Riadov — Senior Engineer at X — Berlin"*).

### Root cause

The profile is built **only** from the body of the fetched page. The rich
`title` and `snippet` that the search step already retrieved are discarded.

Concretely, for a selected LinkedIn candidate the graph runs
`narrow_query` (pick_index) → `fetch_pages` → `build_profile`, and:

1. `fetch_pages` (`app/graph/nodes.py`, `_fetch_and_extract`) fetches the single
   selected URL.
2. `PageFetcher.fetch` checks `robots.txt` first (`app/tools/fetch.py:70`).
   LinkedIn's robots.txt disallows profile scraping for our generic
   `user_agent` (`search4people/0.1`), so `allowed()` returns `False` and
   `markdown` is `""`.
3. Even if robots allowed it, LinkedIn is in `js_heavy_domains`
   (`app/config.py:73`) → Playwright, and a headless fetch without login hits an
   auth-wall ("Join LinkedIn to view…") with no profile content.
4. Empty `markdown` → `_fetch_and_extract` returns `None` → `fetched_pages` is
   empty → `build_profile` (`app/graph/nodes.py:382`) takes the
   `if not partials:` branch and returns `PersonProfile(full_name=...,
   confidence="low")` with all fields empty.

`extract_profile_from_page` (`app/tools/extract.py:29`) only receives
`full_name`, `distinguishers`, `url`, `markdown`, `platform`. The candidate's
`title`/`snippet` never reach the extraction LLM at all.

## Goal

Stop discarding search data. Use the candidate's `title` + `snippet` as a
first-class additional fact source during extraction, so a selected candidate
always yields a populated (if modest) profile — even when the page body is
blocked. No logins, cookies, or paid APIs.

Out of scope: authenticated/3rd-party access to LinkedIn, and any new search to
gather *more* snippets. We only stop losing the data we already have.

## Approach

Chosen: **inject at the extraction layer (per-page)**. The snippet+title flow
into `extract_profile_from_page` for every candidate, so the LLM sees them
alongside the page body and per-source provenance stays intact (each fact bound
to its own URL).

Rejected alternatives:
- **Inject at merge (`build_profile`)** — fewer edit sites, but loses the
  fact→source binding and makes it easier to blend different people.
- **Synthetic page from snippet only on fetch failure** — minimal, but covers
  only the fallback case; we decided the snippet should *always* contribute.

The snippet is used **always** as an additional source, not only as a fallback —
it gives more even extraction quality across all platforms, not just blocked
ones.

## Changes

### 1. `app/tools/extract.py`

- Add two parameters to `extract_profile_from_page`: `title: str | None` and
  `snippet: str | None`.
- In the human message, add a `--- SEARCH RESULT ---` block (title + snippet)
  **before** the `--- PAGE MARKDOWN ---` block.
- Extend the system prompt: the search-result title and snippet are also a
  trustworthy source about this person; if the page body is empty or is a
  login/placeholder page, rely on the search result. Do not fabricate beyond
  what the snippet and page support. Keep the existing "evidence is mandatory,
  cite the URL" rule; populate `evidence.snippet` with the snippet text.

### 2. `app/graph/nodes.py` — `_fetch_and_extract`

Replace the early `return None` on empty markdown with: bail only when there is
**nothing** to extract from (no markdown *and* no snippet/title).

```python
async def _fetch_and_extract(c):
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
        # Bail only if there is nothing at all to extract from.
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

Effect: a robots-/auth-blocked LinkedIn candidate now yields a snippet-based
partial, so `build_profile` sees a non-empty `partials` list and fills the
fields. The empty-profile branch no longer fires for this case.

### Confidence

When the only data is a short snippet, `confidence` honestly stays "low" or
"medium" (decided by the extraction and merge LLMs). The fields (work,
locations, bio, links) are nonetheless populated — which resolves the
"everything is empty" complaint. Low confidence from a single short source is
correct, not a bug.

## Testing

Fast suite (no network, no LLM):

1. `extract.py`: mock `build_structured_model`; assert the message passed to
   `ainvoke` contains the `SEARCH RESULT` block with the title + snippet.
2. `nodes.py::fetch_pages`: mock fetcher returns `FetchResult(markdown="")` for a
   selected candidate that has a snippet; mock `extract_profile_from_page` to
   return a populated `PersonProfile`. Assert `fetched_pages` is non-empty,
   `fetch_blocked is True`, `markdown_len == 0`, and the snippet reached extract.
3. Regression: a candidate with no markdown **and** no snippet/title still
   returns `None` and never enters `fetched_pages`.

Eval (opt-in): add a golden where the "page" is a short LinkedIn snippet (as
`retrieval_context`) and check extraction pulls facts (work/location) without
fabrication — via the existing `tests/evals/data/pages/*.md` + `goldens.json`
mechanism (see DEV.md "Add a new golden").

Post-change checklist (DEV.md): `uv run pytest`, `uv run ruff check .`,
`uv run mypy app`. The graph topology is unchanged, so `test_graph_flow.py` does
not need updating.
