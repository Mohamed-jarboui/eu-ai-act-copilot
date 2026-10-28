"""Gemini provider (google-genai SDK)."""

from __future__ import annotations

import os

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from copilot.llm.base import (
    Embedding,
    Generation,
    ProviderError,
    ServerError,
    Tier,
    error_from_status,
)

# Overridable via the constructor. Verify against the providers' current model lists.
DEFAULT_MODELS: dict[Tier, str] = {"small": "gemini-2.5-flash-lite", "large": "gemini-2.5-flash"}
DEFAULT_EMBED_MODEL = "gemini-embedding-001"


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        models: dict[Tier, str] | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=key)
        self._models = models or DEFAULT_MODELS
        self._embed_model = embed_model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Generation:
        model = self._models[tier]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            response = self._client.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except Exception as exc:
            raise _translate(exc) from exc
        meta = response.usage_metadata
        return Generation(
            text=response.text or "",
            model=model,
            input_tokens=meta.prompt_token_count if meta else None,
            output_tokens=meta.candidates_token_count if meta else None,
        )

    def embed(self, texts: list[str]) -> Embedding:
        try:
            response = self._client.models.embed_content(model=self._embed_model, contents=texts)
        except Exception as exc:
            raise _translate(exc) from exc
        return Embedding(
            vectors=[list(e.values or []) for e in response.embeddings or []],
            model=self._embed_model,
        )


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, genai_errors.APIError):
        return error_from_status(exc.code, str(exc), provider=GeminiProvider.name)
    if isinstance(exc, httpx.TransportError | TimeoutError | ConnectionError):
        return ServerError(str(exc), provider=GeminiProvider.name)
    raise exc
