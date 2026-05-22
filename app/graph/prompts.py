"""Prompts used by graph nodes.

Kept in English regardless of UI locale; the LLM is steered to answer in the
user's locale only when its output is shown directly in the chat (the
`ask_question` helper renders user-facing copy via the i18n table instead).
"""

from __future__ import annotations

SYSTEM_RESEARCHER = """You are an OSINT-style people-search assistant operating
inside a multi-step LangGraph workflow.

Hard rules:
- Public information only. Never invent details. Cite sources for every claim.
- When evidence is thin, prefer "unknown" over guessing.
- Be culturally neutral: a name shared by many people is normal; ask narrowing
  questions before committing to a profile.
"""

NARROW_QUERY_PROMPT = """We currently have multiple plausible candidates for
the person below.

Candidates (top 8 shown, with platform, title, snippet):
{candidates}

Already-collected identifying attributes:
{known_attributes}

Pick the SINGLE most discriminating attribute the user could still provide
(choose from: age, country, city, school, university, employer, profession,
distinctive_event). Phrase one short question asking for it.

Additionally extract up to 5 candidate VALUES for that attribute that actually
appear in the candidates' titles/snippets above (e.g., if `attribute` is
"employer", list the company names you can see in the snippets; if "city", list
the cities). Use only values literally observable in the candidate list — do
not invent. If you cannot extract any, return an empty list.

Respond with JSON ONLY in this shape:
{{
  "attribute": "<one of the names above>",
  "question_en": "...",
  "question_ru": "...",
  "options": ["value1", "value2", ...]
}}
"""

DISAMBIGUATE_PROMPT = """We still have multiple plausible candidates after
narrowing. Pick ONE concrete, verifiable fact about ONE of the candidates that
the user can confirm or deny (e.g. "Did this person speak at PyCon 2023?",
"Does this person's GitHub bio mention Rust?").

Candidates:
{candidates}

Respond with JSON ONLY:
{{"candidate_url": "...", "fact_en": "...", "fact_ru": "..."}}
"""

MERGE_PROFILE_PROMPT = """Merge the partial profiles extracted from individual
pages into one coherent PersonProfile.

Rules:
- Prefer the most specific value when sources agree.
- If sources disagree, keep the value from the higher-authority source
  (LinkedIn/GitHub > X/Twitter > Facebook/Instagram/VK > generic web).
- Deduplicate education/work/links by (organization, year-range) / URL.
- Populate `evidence` with every URL that contributed anything.
- Set `confidence` based on how many independent sources agree: high (3+),
  medium (2), low (1).

Person we are profiling: {full_name}
Distinguishing attributes: {distinguishers}

Partial profiles to merge:
{partials}
"""
