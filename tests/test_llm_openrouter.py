"""OpenRouter provider with a mocked HTTP transport (no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from copilot.llm import BadRequestError, RateLimitError, ServerError
from copilot.llm.openrouter import DEFAULT_MODELS, OpenRouterProvider


def provider(handler) -> OpenRouterProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenRouterProvider(api_key="test-key", client=client, reasoning_effort="medium")


def reply(text="ok", status=200, **extra) -> httpx.Response:
    body = {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}, **extra}  # fmt: skip
    return httpx.Response(status, json=body)


def test_generate_sends_the_model_prompt_and_key_and_reads_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return reply('{"verdict": 1}')

    out = provider(handler).generate("judge this", system="be strict", max_output_tokens=50)
    assert out.text == '{"verdict": 1}' and out.model == DEFAULT_MODELS["large"]
    assert (out.input_tokens, out.output_tokens) == (12, 3)
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == DEFAULT_MODELS["large"] and body["max_tokens"] == 50
    assert body["messages"][0] == {"role": "system", "content": "be strict"}
    assert body["reasoning"] == {"effort": "medium"} and body["temperature"] == 0.0


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(429, text="rate limited"), RateLimitError),
        (httpx.Response(503, text="down"), ServerError),
        (httpx.Response(400, text="bad"), BadRequestError),
        (httpx.Response(200, json={"error": {"code": 429, "message": "upstream"}}), RateLimitError),
        (httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}), ServerError),
    ],
)
def test_errors_map_to_provider_errors(response, error):
    with pytest.raises(error):
        provider(lambda request: response).generate("x")


def test_connection_failure_is_a_server_error():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(ServerError):
        provider(handler).generate("x")


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterProvider()
