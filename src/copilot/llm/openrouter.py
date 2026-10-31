"""OpenRouter provider: one OpenAI-compatible endpoint for many model families.

Used for the evaluation judge, so the judge comes from a different model family than the agent
(Gemini and gpt-oss). Plain httpx (already installed with the groq SDK) instead of a new SDK.
Only generate() is offered: the judge needs neither streaming nor embeddings.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx

from copilot.llm.base import (
    Embedding,
    EmbedTask,
    Generation,
    ServerError,
    StreamEvent,
    Tier,
    UnsupportedOperationError,
    error_from_status,
)

URL = "https://openrouter.ai/api/v1/chat/completions"
# Free models listed by https://openrouter.ai/api/v1/models on 2026-10-31 (the Kimi models the
# plan named are no longer free). Nemotron 3 Ultra: 550B mixture-of-experts, 55B active.
DEFAULT_MODELS: dict[Tier, str] = {
    "small": "google/gemma-4-31b-it:free",
    "large": "nvidia/nemotron-3-ultra-550b-a55b:free",
}


class OpenRouterProvider:
    name = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        models: dict[Tier, str] | None = None,
        reasoning_effort: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._client = client or httpx.Client(timeout=timeout_s)
        self._models = models or DEFAULT_MODELS
        self._extra = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}

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
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages, "temperature": temperature, **self._extra}
        if max_output_tokens:
            payload["max_tokens"] = max_output_tokens
        try:
            response = self._client.post(URL, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:  # connection problems and timeouts
            raise ServerError(str(exc), provider=self.name) from exc
        if response.status_code != 200:
            raise error_from_status(response.status_code, response.text[:500], provider=self.name)
        data = response.json()
        if "error" in data:  # OpenRouter can report an upstream failure inside a 200
            error = data["error"] if isinstance(data["error"], dict) else {}
            code = error.get("code") if isinstance(error.get("code"), int) else 502
            raise error_from_status(code, str(data["error"])[:500], provider=self.name)
        choices = data.get("choices") or []
        text = (choices[0].get("message") or {}).get("content") if choices else None
        if not text:
            raise ServerError("empty reply", provider=self.name)
        usage = data.get("usage") or {}
        return Generation(
            text=text,
            model=model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Iterator[StreamEvent]:
        raise UnsupportedOperationError(
            "streaming is not implemented for OpenRouter", provider=self.name
        )

    def embed(
        self, texts: list[str], *, task: EmbedTask = "document", dimensions: int | None = None
    ) -> Embedding:
        raise UnsupportedOperationError("OpenRouter embeddings are not used", provider=self.name)
