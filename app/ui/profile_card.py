"""Chainlit rendering for a PersonProfile."""

from __future__ import annotations

import json

import chainlit as cl

from app.models.profile import PersonProfile


def render_profile_message(profile: PersonProfile, locale: str = "en") -> cl.Message:
    """Build a Chainlit message with the profile body + a JSON attachment."""
    body = profile.as_markdown()
    elements: list[cl.Element] = [
        cl.File(
            name=f"{profile.full_name.replace(' ', '_')}.json",
            content=json.dumps(
                profile.model_dump(mode="json"), ensure_ascii=False, indent=2
            ).encode("utf-8"),
            mime="application/json",
            display="inline",
        )
    ]
    if profile.photo_url:
        elements.insert(
            0,
            cl.Image(
                name=profile.full_name,
                url=str(profile.photo_url),
                display="inline",
            ),
        )
    title = "Profile" if locale == "en" else "Профиль"
    return cl.Message(content=f"### {title}\n\n{body}", elements=elements)
