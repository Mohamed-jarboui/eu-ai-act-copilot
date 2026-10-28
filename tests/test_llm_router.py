"""Router behaviour with fake providers: no network, no real sleeping."""

import pytest

from copilot.llm import (
    AllProvidersFailedError,
    BadRequestError,
    Embedding,
    FallbackLLM,
    Generation,
    RateLimitError,
    ServerError,
    UnsupportedOperationError,
)


class FakeProvider:
    """Plays back a script: each call pops one item; an exception instance is raised."""

    def __init__(self, name: str, script: list, embed_script: list | None = None) -> None:
        self.name = name
        self.script = list(script)
        self.embed_script = list(embed_script or [])
        self.calls = 0

    def generate(self, prompt, **kwargs) -> Generation:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return Generation(text=item, model=f"{self.name}-model", input_tokens=7, output_tokens=3)

    def embed(self, texts) -> Embedding:
        self.calls += 1
        item = self.embed_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return Embedding(vectors=[[0.1, 0.2] for _ in texts], model=f"{self.name}-embed")


def make_llm(*providers, max_attempts=3):
    sleeps: list[float] = []
    ticks = iter(range(0, 1000))  # fake clock: every reading advances 1s, so latency is 1.0
    llm = FallbackLLM(
        list(providers),
        max_attempts=max_attempts,
        backoff_base_s=1.0,
        sleep=sleeps.append,
        clock=lambda: float(next(ticks)),
    )
    return llm, sleeps


def test_first_provider_success_makes_no_fallback():
    primary = FakeProvider("gemini", ["hello"])
    backup = FakeProvider("groq", ["unused"])
    llm, _ = make_llm(primary, backup)

    assert llm.generate("hi").text == "hello"
    assert backup.calls == 0


def test_fallback_to_second_provider_after_rate_limit():
    primary = FakeProvider("gemini", [RateLimitError("429")] * 3)
    backup = FakeProvider("groq", ["from groq"])
    llm, _ = make_llm(primary, backup)

    result = llm.generate("hi")

    assert result.text == "from groq"
    assert primary.calls == 3  # retried up to max_attempts before giving up
    assert backup.calls == 1


def test_transient_error_is_retried_on_same_provider_with_backoff():
    primary = FakeProvider("gemini", [ServerError("503"), ServerError("503"), "ok"])
    backup = FakeProvider("groq", ["unused"])
    llm, sleeps = make_llm(primary, backup)

    assert llm.generate("hi").text == "ok"
    assert backup.calls == 0
    assert sleeps == [1.0, 2.0]  # exponential: base * 2**attempt


def test_retries_stop_at_max_attempts():
    primary = FakeProvider("gemini", [ServerError("503")] * 10)
    llm, sleeps = make_llm(primary, max_attempts=2)

    with pytest.raises(AllProvidersFailedError):
        llm.generate("hi")

    assert primary.calls == 2
    assert len(sleeps) == 1  # no sleep after the final attempt


def test_bad_request_is_neither_retried_nor_sent_to_fallback():
    primary = FakeProvider("gemini", [BadRequestError("bad prompt")])
    backup = FakeProvider("groq", ["unused"])
    llm, sleeps = make_llm(primary, backup)

    with pytest.raises(BadRequestError):
        llm.generate("hi")

    assert primary.calls == 1
    assert backup.calls == 0
    assert sleeps == []


def test_usage_recorded_for_every_attempt_including_failures():
    primary = FakeProvider("gemini", [RateLimitError("429")] * 3)
    backup = FakeProvider("groq", ["from groq"])
    llm, _ = make_llm(primary, backup)

    llm.generate("hi")

    assert [(u.provider, u.error is None) for u in llm.usage] == [
        ("gemini", False),
        ("gemini", False),
        ("gemini", False),
        ("groq", True),
    ]
    final = llm.usage[-1]
    assert (final.model, final.input_tokens, final.output_tokens) == ("groq-model", 7, 3)
    assert final.latency_s == 1.0
    assert final.operation == "generate"


def test_embed_skips_provider_without_embeddings():
    primary = FakeProvider("groq", [], embed_script=[UnsupportedOperationError("no embeddings")])
    backup = FakeProvider("gemini", [], embed_script=[[]])
    llm, _ = make_llm(primary, backup)

    result = llm.embed(["a", "b"])

    assert len(result.vectors) == 2
    assert [u.provider for u in llm.usage] == ["gemini"]  # skipped provider leaves no record
    assert llm.usage[0].operation == "embed"


def test_all_providers_failing_raises():
    llm, _ = make_llm(
        FakeProvider("gemini", [ServerError("500")] * 3),
        FakeProvider("groq", [ServerError("500")] * 3),
    )

    with pytest.raises(AllProvidersFailedError):
        llm.generate("hi")
