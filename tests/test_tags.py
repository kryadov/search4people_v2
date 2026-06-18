"""Auto-tag computation + searchable tag-line rendering."""

from __future__ import annotations

from app.ui.tags import compute_auto_tags, render_tag_line


def _profile(**over):
    base = {
        "full_name": "Jane Doe",
        "confidence": "high",
        "evidence": [
            {"url": "https://github.com/jane", "platform": "github"},
            {"url": "https://linkedin.com/in/jane", "platform": "linkedin"},
            {"url": "https://github.com/jane2", "platform": "github"},  # dup
            {"url": "https://example.com/x", "platform": None},  # no platform
        ],
    }
    base.update(over)
    return base


def test_compute_auto_tags_dedups_platforms_and_adds_confidence_locale() -> None:
    tags = compute_auto_tags(_profile(), "ru")
    assert tags == [
        "platform:github",
        "platform:linkedin",
        "confidence:high",
        "locale:ru",
    ]


def test_compute_auto_tags_defaults_confidence_when_missing() -> None:
    tags = compute_auto_tags({"full_name": "X", "evidence": []}, "en")
    assert tags == ["confidence:medium", "locale:en"]


def test_render_tag_line_includes_label_and_tokens() -> None:
    line = render_tag_line(["platform:github", "confidence:high"], "en")
    assert "Tags:" in line
    assert "platform:github" in line
    assert "confidence:high" in line


def test_render_tag_line_is_localized() -> None:
    line = render_tag_line(["locale:ru"], "ru")
    assert "Теги:" in line
