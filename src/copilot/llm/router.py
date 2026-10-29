"""FallbackLLM: try providers in order, retry transient errors with backoff, record usage."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import TypeVar

from copilot.llm.base import (
    AllProvidersFailedError,
    Embedding,
    EmbedTask,
    Generation,
    Provider,
    ProviderError,
    RateLimitError,
    ServerError,
    StreamEvent,
    Tier,
    UnsupportedOperationError,
    Usage,
)

T = TypeVar("T")


def is_retryable(error: ProviderError) -> bool:
    """Errors worth retrying on the SAME provider: the next attempt may well succeed."""
    return isinstance(error, RateLimitError | ServerError)


def should_fallback(error: ProviderError) -> bool:
    """Decide whether to move on to the next provider after `error`.

    Called once a provider is finished with (its retries are used up, or the error was not
    retryable). Return True to try the next provider, False to stop and raise.
    """
    # TODO(me): decide which errors trigger fallback to the next provider.
    # Hint: ask "would the NEXT provider plausibly succeed with the same request?"
    #   - RateLimitError / ServerError: the problem is on this provider's side -> usually yes.
    #   - BadRequestError: the request itself is wrong -> would Groq accept it? Probably not.
    #   - AuthError: our Gemini key is bad. Groq has its own key... so what is the right call?
    #     Think about whether failing loudly or degrading silently is better for debugging.
    # Placeholder below is conservative so the tests can run; replace it with your own logic.
    return isinstance(error, RateLimitError | ServerError)


class FallbackLLM:
    """Same generate()/embed() surface as a Provider, backed by an ordered list of providers."""

    def __init__(
        self,
        providers: list[Provider],
        *,
        max_attempts: int = 3,
        backoff_base_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not providers:
            raise ValueError("FallbackLLM needs at least one provider")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.providers = providers
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep
        self._clock = clock
        self.usage: list[Usage] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Generation:
        def call(provider: Provider) -> Generation:
            return provider.generate(
                prompt,
                system=system,
                tier=tier,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

        def usage_of(provider: Provider, result: Generation | None, t: float, err: str | None):
            return Usage(
                provider=provider.name,
                model=result.model if result else "",
                operation="generate",
                input_tokens=result.input_tokens if result else None,
                output_tokens=result.output_tokens if result else None,
                latency_s=t,
                error=err,
            )

        return self._run(call, usage_of)

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream text deltas, then a final event with the totals.

        Retries and fallback apply only until the first event arrives (connection, quota and
        auth errors surface there). An error after text has started is raised to the caller:
        switching provider mid-answer would repeat or contradict what was already shown.
        """

        def call(provider: Provider):
            if not hasattr(provider, "generate_stream"):
                raise UnsupportedOperationError("no streaming", provider=provider.name)
            events = provider.generate_stream(
                prompt,
                system=system,
                tier=tier,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            return next(events, None), events  # pulls the first event so early errors raise here

        def usage_of(provider: Provider, result, t: float, err: str | None) -> Usage:
            return Usage(provider.name, "", "generate", None, None, t, err)

        start = self._clock()
        first, events = self._run(call, usage_of)
        index = len(self.usage) - 1  # the record _run just appended for the successful attempt
        last = first
        if first is not None:
            yield first
        for event in events:
            last = event
            yield event
        if last is not None and last.final is not None:
            record = self.usage[index]
            self.usage[index] = Usage(
                record.provider,
                last.final.model,
                "generate",
                last.final.input_tokens,
                last.final.output_tokens,
                self._clock() - start,
                None,
            )

    def embed(
        self, texts: list[str], *, task: EmbedTask = "document", dimensions: int | None = None
    ) -> Embedding:
        def call(provider: Provider) -> Embedding:
            return provider.embed(texts, task=task, dimensions=dimensions)

        def usage_of(provider: Provider, result: Embedding | None, t: float, err: str | None):
            return Usage(
                provider=provider.name,
                model=result.model if result else "",
                operation="embed",
                input_tokens=result.input_tokens if result else None,
                output_tokens=None,
                latency_s=t,
                error=err,
            )

        return self._run(call, usage_of)

    def _run(
        self,
        call: Callable[[Provider], T],
        usage_of: Callable[[Provider, T | None, float, str | None], Usage],
    ) -> T:
        last_error: ProviderError | None = None
        for provider in self.providers:
            for attempt in range(self.max_attempts):
                start = self._clock()
                try:
                    result = call(provider)
                except UnsupportedOperationError as exc:
                    # Not a failure: this provider simply cannot do it. Skip, no usage record.
                    last_error = exc
                    break
                except ProviderError as exc:
                    elapsed = self._clock() - start
                    self.usage.append(usage_of(provider, None, elapsed, repr(exc)))
                    last_error = exc
                    if is_retryable(exc) and attempt < self.max_attempts - 1:
                        self._sleep(self.backoff_base_s * 2**attempt)
                        continue
                    break
                else:
                    elapsed = self._clock() - start
                    self.usage.append(usage_of(provider, result, elapsed, None))
                    return result
            if last_error is not None and not isinstance(last_error, UnsupportedOperationError):
                if not should_fallback(last_error):
                    raise last_error
        raise AllProvidersFailedError(
            f"all providers failed; last error: {last_error!r}",
            provider=last_error.provider if last_error else "",
        )
