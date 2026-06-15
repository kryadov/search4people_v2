"""Public guardrails facade.

Imports are deferred into `get_guardrails()` so that importing this package (which
`app.config` does, to reach `app.guardrails.config`) does not pull `app.config`
back in at module-load time and create a circular import.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.guardrails.backends.base import GuardBackend
    from app.guardrails.engine import Guardrails, NoOpGuardrails


@lru_cache(maxsize=1)
def get_guardrails() -> Guardrails | NoOpGuardrails:
    from app.config import get_settings
    from app.guardrails.engine import Guardrails, NoOpGuardrails

    g = get_settings().guardrails
    if not g.enabled or g.backend == "noop":
        return NoOpGuardrails()
    backend: GuardBackend
    if g.backend == "http":
        from app.guardrails.backends.http import HttpBackend

        backend = HttpBackend(g.http_url or "")
    else:
        from app.guardrails.backends.local import LocalGlinerBackend

        backend = LocalGlinerBackend(g)
    return Guardrails(backend=backend, settings=g)
