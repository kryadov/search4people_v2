"""Translate LangGraph stream events into Chainlit `cl.Step` updates."""

from __future__ import annotations

from typing import Any

import chainlit as cl

# Map LangGraph node names to short human-friendly status labels per locale.
_NODE_LABELS: dict[str, dict[str, str]] = {
    "collect_identity": {"en": "Collecting identity", "ru": "Собираю входные данные"},
    "preliminary_search": {"en": "Preliminary search", "ru": "Предварительный поиск"},
    "evaluate_candidates": {"en": "Evaluating candidates", "ru": "Оцениваю кандидатов"},
    "narrow_query": {"en": "Narrowing the query", "ru": "Сужаю запрос"},
    "expand_search": {"en": "Expanding the search", "ru": "Расширяю поиск"},
    "fetch_pages": {"en": "Fetching pages", "ru": "Извлекаю страницы"},
    "build_profile": {"en": "Building profile", "ru": "Собираю профиль"},
    "confirm_profile": {"en": "Awaiting confirmation", "ru": "Жду подтверждения"},
}


def label_for(node: str, locale: str) -> str:
    bundle = _NODE_LABELS.get(node)
    if not bundle:
        return node
    return bundle.get(locale) or bundle.get("en") or node


async def emit_node_step(node: str, locale: str, output: Any | None = None) -> None:
    """Emit a transient Chainlit step for a finished graph node."""
    label = label_for(node, locale)
    async with cl.Step(name=label, type="tool") as step:
        if output is not None:
            try:
                if isinstance(output, dict):
                    summary_keys = [k for k in ("phase", "iteration") if k in output]
                    if "candidates" in output:
                        summary_keys.append("candidates")
                    summary = {
                        k: (len(output[k]) if k == "candidates" else output[k])
                        for k in summary_keys
                    }
                    step.output = summary or "ok"
                else:
                    step.output = "ok"
            except Exception:
                step.output = "ok"
