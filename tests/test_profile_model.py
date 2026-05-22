"""Validate the PersonProfile schema renders sensibly and is round-trippable."""

from __future__ import annotations

from app.models.profile import Education, Evidence, Link, PersonProfile, Work


def test_minimum_profile_renders():
    p = PersonProfile(full_name="Jane Doe")
    md = p.as_markdown()
    assert "Jane Doe" in md
    assert "Confidence" in md


def test_full_profile_round_trip():
    p = PersonProfile(
        full_name="John Smith",
        aliases=["JS"],
        date_of_birth="1985",
        education=[
            Education(institution="MIT", degree="BSc", field="CS", start_year=2003, end_year=2007)
        ],
        work=[Work(organization="Acme", title="Engineer", start_year=2008)],
        links=[Link(platform="github", url="https://github.com/jsmith", handle="jsmith")],
        evidence=[Evidence(url="https://example.com/jsmith", platform="web", snippet="hit")],
        confidence="high",
    )
    dumped = p.model_dump(mode="json")
    restored = PersonProfile.model_validate(dumped)
    assert restored.full_name == "John Smith"
    assert restored.education[0].institution == "MIT"
    assert restored.links[0].handle == "jsmith"
    assert str(restored.evidence[0].url).startswith("https://example.com")
    md = restored.as_markdown()
    assert "MIT" in md and "Acme" in md and "github" in md
