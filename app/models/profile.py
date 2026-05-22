"""Structured profile schema used as the final agent output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class Location(BaseModel):
    country: str | None = None
    city: str | None = None
    region: str | None = None
    notes: str | None = None


class Education(BaseModel):
    institution: str
    degree: str | None = None
    field: str | None = None
    start_year: int | None = None
    end_year: int | None = None


class Work(BaseModel):
    organization: str
    title: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    location: str | None = None


class Link(BaseModel):
    platform: str
    url: HttpUrl
    handle: str | None = None


class Evidence(BaseModel):
    url: HttpUrl
    platform: str | None = None
    snippet: str | None = None
    supports: list[str] = Field(
        default_factory=list,
        description="Names of profile fields this evidence backs.",
    )


class Candidate(BaseModel):
    """Lightweight pre-merge representation of a single search hit."""

    url: HttpUrl
    title: str | None = None
    snippet: str | None = None
    platform: str | None = None
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Coarse 0..1 confidence that this hit refers to the target person.",
    )


Confidence = Literal["low", "medium", "high"]


class PersonProfile(BaseModel):
    """The structured final profile shown to the user and persisted to SQLite."""

    full_name: str
    given_names: list[str] = Field(default_factory=list)
    family_names: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    date_of_birth: str | None = Field(
        default=None,
        description="ISO-8601 date or year if only year is known.",
    )
    locations: list[Location] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    work: list[Work] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    bio: str | None = None
    photo_url: HttpUrl | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence = "medium"

    def as_markdown(self) -> str:
        """Render a human-readable Markdown summary."""
        lines: list[str] = [f"### {self.full_name}", ""]
        if self.aliases:
            lines.append(f"**Also known as:** {', '.join(self.aliases)}")
        if self.date_of_birth:
            lines.append(f"**Date of birth:** {self.date_of_birth}")
        if self.locations:
            loc_strs = [
                ", ".join(filter(None, [loc.city, loc.region, loc.country])) or "—"
                for loc in self.locations
            ]
            lines.append(f"**Locations:** {' / '.join(loc_strs)}")
        if self.education:
            lines.append("\n**Education:**")
            for ed in self.education:
                period = f"{ed.start_year or ''}–{ed.end_year or ''}".strip("–")
                lines.append(
                    f"- {ed.institution}"
                    + (f", {ed.degree}" if ed.degree else "")
                    + (f" ({ed.field})" if ed.field else "")
                    + (f" [{period}]" if period else "")
                )
        if self.work:
            lines.append("\n**Work:**")
            for w in self.work:
                period = f"{w.start_year or ''}–{w.end_year or ''}".strip("–")
                lines.append(
                    f"- {w.title or ''} at **{w.organization}**".strip()
                    + (f" [{period}]" if period else "")
                    + (f" — {w.location}" if w.location else "")
                )
        if self.links:
            lines.append("\n**Links:**")
            for link in self.links:
                handle = f" (@{link.handle})" if link.handle else ""
                lines.append(f"- [{link.platform}{handle}]({link.url})")
        if self.bio:
            lines.append(f"\n**Bio:** {self.bio}")
        lines.append(f"\n_Confidence: **{self.confidence}**_")
        if self.evidence:
            lines.append("\n**Sources:**")
            for ev in self.evidence:
                lines.append(f"- <{ev.url}>" + (f" — {ev.snippet}" if ev.snippet else ""))
        return "\n".join(lines)
