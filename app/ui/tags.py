"""Automatic per-search thread tags.

Tags are namespaced tokens (`platform:github`, `confidence:high`, `locale:ru`)
so the built-in Chainlit search can find threads precisely. They are stored in
thread metadata and echoed as a searchable line in the conversation.
"""

from __future__ import annotations

from typing import Any

from app.i18n import Locale, t


def compute_auto_tags(profile: dict[str, Any], locale: str) -> list[str]:
    """Derive deduped, namespaced tags from a built profile + conversation locale."""
    tags: list[str] = []
    seen: set[str] = set()
    for ev in profile.get("evidence") or []:
        platform = (ev.get("platform") or "").strip().lower()
        if not platform:
            continue
        tag = f"platform:{platform}"
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    confidence = profile.get("confidence") or "medium"
    tags.append(f"confidence:{confidence}")
    tags.append(f"locale:{locale}")
    return tags


def render_tag_line(tags: list[str], locale: Locale) -> str:
    """A compact, searchable one-liner echoing the tags into the thread."""
    label = t("tags_line_label", locale)
    return f"🏷 {label} " + " · ".join(tags)
