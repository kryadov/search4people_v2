"""Application configuration loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai", "ollama"]
SearchProvider = Literal["tavily", "ddg"]


def _split_csv(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: LLMProvider = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # Search
    search_providers: Annotated[list[SearchProvider], NoDecode] = Field(
        default_factory=lambda: ["tavily", "ddg"]  # type: ignore[arg-type]
    )
    tavily_api_key: str | None = None

    # Platforms
    platforms_primary: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["linkedin", "github", "twitter"]
    )
    platforms_secondary: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["facebook", "instagram", "vk"]
    )

    # Graph
    max_iterations: int = 7

    # Scraping
    per_host_rps: float = 1.0
    user_agent: str = "search4people/0.1 (+https://github.com/your-org/search4people_v2)"
    js_heavy_domains: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["linkedin.com", "instagram.com", "facebook.com"]
    )

    # Storage
    db_path: Path = Path("data/app.db")

    # Chainlit
    chainlit_auth_secret: str = "replace-me-with-a-long-random-secret"

    # Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "search4people"

    @field_validator(
        "search_providers",
        "platforms_primary",
        "platforms_secondary",
        "js_heavy_domains",
        mode="before",
    )
    @classmethod
    def _parse_csv(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return _split_csv(value)
        if isinstance(value, list):
            return _split_csv(value)
        raise TypeError(f"Expected list or comma-separated string, got {type(value)!r}")

    @property
    def all_platforms(self) -> list[str]:
        return [*self.platforms_primary, *self.platforms_secondary]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
