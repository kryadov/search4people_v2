"""LLM provider factory.

Wraps `langchain.chat_models.init_chat_model` so the rest of the app picks a
provider through `Settings` rather than importing provider-specific classes.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import Settings, get_settings


def _provider_kwargs(settings: Settings) -> dict[str, Any]:
    """Map our Settings into the kwargs init_chat_model expects per provider."""
    provider = settings.llm_provider
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the anthropic provider")
        return {"api_key": settings.anthropic_api_key}
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai provider")
        return {"api_key": settings.openai_api_key}
    if provider == "ollama":
        return {
            "base_url": settings.ollama_base_url,
            # Keep the model resident between calls to avoid the unload/reload
            # race that can break JSON-schema (format) requests on some models.
            "keep_alive": settings.ollama_keep_alive,
        }
    raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache(maxsize=4)
def build_chat_model(temperature: float = 0.0) -> BaseChatModel:
    """Return a chat model built from the active settings.

    Cached because LangChain providers create their own HTTP clients; reusing a
    single instance avoids socket churn across graph node invocations.
    """
    settings = get_settings()
    kwargs = _provider_kwargs(settings)
    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=temperature,
        **kwargs,
    )


def build_structured_model(schema: type, temperature: float = 0.0) -> Any:
    """Return a chat model bound to `schema` for structured output.

    On Ollama the default strict JSON-schema grammar can fail to load the model
    vocabulary for some models (e.g. gpt-oss), so we use the configured method
    (`ollama_structured_output_method`, default "function_calling"). Cloud
    providers keep LangChain's default method.
    """
    settings = get_settings()
    model = build_chat_model(temperature)
    if settings.llm_provider == "ollama":
        return model.with_structured_output(
            schema, method=settings.ollama_structured_output_method
        )
    return model.with_structured_output(schema)
