"""Groq provider (groq SDK). Groq has no embeddings endpoint, so embed() is unsupported."""

from __future__ import annotations

import os

import groq as groq_sdk

from copilot.llm.base import (
    Embedding,
    Generation,
    ProviderError,
    ServerError,
    Tier,
    UnsupportedOperationError,
    error_from_status,
)

# Overridable via the constructor. Verify against the providers' current model lists.
DEFAULT_MODELS: dict[Tier, str] = {
    "small": "llama-3.1-8b-instant",
    "large": "llama-3.3-70b-versatile",
}


class GroqProvider:
    name = "groq"

    def __init__(
        self, api_key: str | None = None, *, models: dict[Tier, str] | None = None
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set")
        self._client = groq_sdk.Groq(api_key=key, max_retries=0)  # retries are the router's job
        self._models = models or DEFAULT_MODELS

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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
        except Exception as exc:
            raise _translate(exc) from exc
        usage = response.usage
        return Generation(
            text=response.choices[0].message.content or "",
            model=model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )

    def embed(self, texts: list[str]) -> Embedding:
        raise UnsupportedOperationError("Groq has no embeddings endpoint", provider=self.name)


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, groq_sdk.APIStatusError):
        return error_from_status(exc.status_code, str(exc), provider=GroqProvider.name)
    if isinstance(exc, groq_sdk.APIConnectionError):  # includes APITimeoutError
        return ServerError(str(exc), provider=GroqProvider.name)
    raise exc
