"""DeepEval judge that reuses the application's configured chat model.

Routing the LLM-as-judge through `app.llm.build_chat_model` means evals use the
same provider/model as the app — by default local Ollama + gpt-oss, no API key.
"""

from __future__ import annotations

from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

from app.config import get_settings
from app.llm import build_chat_model


class LangChainJudge(DeepEvalBaseLLM):
    """Adapter exposing the app's chat model through DeepEval's LLM interface."""

    def __init__(self, temperature: float = 0.0) -> None:
        self._temperature = temperature
        self._model: Any = None

    def load_model(self) -> Any:
        if self._model is None:
            self._model = build_chat_model(temperature=self._temperature)
        return self._model

    def get_model_name(self) -> str:
        return f"app-llm:{get_settings().llm_model}"

    def _structured(self, schema: type[BaseModel]) -> Any:
        # Mirror app.llm.build_structured_model: on Ollama the strict JSON-schema
        # grammar can fail to load the vocabulary for some models, so use the
        # configured method (default function_calling).
        model = self.load_model()
        settings = get_settings()
        if settings.llm_provider == "ollama":
            return model.with_structured_output(
                schema, method=settings.ollama_structured_output_method
            )
        return model.with_structured_output(schema)

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        # DeepEval 4.x passes a Pydantic `schema` for its structured metric steps
        # and expects an instance of it back, so we must NOT fall back to plain
        # text here — let any structured-output error propagate with its message.
        if schema is not None:
            return self._structured(schema).invoke(prompt)
        result = self.load_model().invoke(prompt)
        return getattr(result, "content", str(result))

    async def a_generate(
        self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any
    ) -> Any:
        if schema is not None:
            return await self._structured(schema).ainvoke(prompt)
        result = await self.load_model().ainvoke(prompt)
        return getattr(result, "content", str(result))
