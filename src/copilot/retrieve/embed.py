"""Embed texts through the provider layer, in batches, with a disk cache."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from copilot.llm.base import AllProvidersFailedError, EmbedTask
from copilot.retrieve.cache import DiskCache, digest

EMBED_MODEL = "gemini-embedding-001"
DIMENSIONS = 768
BATCH_SIZE = 100
MAX_EMBED_CHARS = 6000  # longest chunk is ~7000 chars; the model limit is in tokens, so stay safe


class QuotaExhaustedError(Exception):
    """A per-day quota is spent: retrying in a few minutes cannot succeed."""


@dataclass
class EmbedStats:
    """What an embedding job cost (cache hits are free, and counted separately)."""

    texts: int = 0
    cache_hits: int = 0
    calls: int = 0
    chars: int = 0
    api_seconds: float = 0.0
    truncated: int = 0


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class Embedder:
    """`llm` is anything with `.embed(texts, task=, dimensions=)` (the FallbackLLM)."""

    def __init__(
        self,
        llm,
        cache: DiskCache,
        *,
        model: str = EMBED_MODEL,
        dimensions: int = DIMENSIONS,
        batch_size: int = BATCH_SIZE,
        retry_wait_s: float = 20.0,
        max_waits: int = 10,
        sleep: Callable[[float], None] = time.sleep,
        progress: Callable[[str], None] = lambda _msg: None,
    ) -> None:
        self.llm, self.cache, self.model, self.dimensions = llm, cache, model, dimensions
        self.batch_size, self.retry_wait_s, self.max_waits = batch_size, retry_wait_s, max_waits
        self._sleep, self._progress = sleep, progress

    def _key(self, text: str, task: EmbedTask) -> str:
        return digest(self.model, task, self.dimensions, text)

    def cached(self, texts: list[str], task: EmbedTask) -> np.ndarray | None:
        """Unit-length vectors for `texts` if every one is in the disk cache, else None."""
        rows = [self.cache.get_embedding(self._key(t[:MAX_EMBED_CHARS], task)) for t in texts]
        if any(row is None for row in rows):
            return None
        return _normalise(np.stack([row[0] for row in rows]))

    def embed(
        self, texts: list[str], task: EmbedTask, stats: EmbedStats | None = None
    ) -> tuple[np.ndarray, list[float]]:
        """Return (unit-length vectors, one recorded API latency per text); cached texts cost 0."""
        stats = stats if stats is not None else EmbedStats()
        stats.truncated += sum(1 for t in texts if len(t) > MAX_EMBED_CHARS)
        texts = [t[:MAX_EMBED_CHARS] for t in texts]
        keys = [self._key(t, task) for t in texts]
        found = {k: self.cache.get_embedding(k) for k in set(keys)}
        missing = [
            (k, t) for k, t in dict(zip(keys, texts, strict=True)).items() if found[k] is None
        ]
        stats.texts += len(texts)
        stats.cache_hits += len(texts) - len(missing)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            vectors, latency = self._call([t for _, t in batch], task)
            stats.calls += 1
            stats.chars += sum(len(t) for _, t in batch)
            stats.api_seconds += latency
            share = latency / len(batch)  # a batch is one request: split its latency across texts
            self.cache.put_embeddings(
                [(k, v, share, len(t)) for (k, t), v in zip(batch, vectors, strict=True)]
            )
            self._progress(f"embedded {start + len(batch)}/{len(missing)} ({task})")
        rows = [found[k] or self.cache.get_embedding(k) for k in keys]
        vectors = _normalise(np.stack([r[0] for r in rows]))
        return vectors, [r[1] for r in rows]

    def _call(self, texts: list[str], task: EmbedTask) -> tuple[np.ndarray, float]:
        for attempt in range(self.max_waits + 1):
            start = time.perf_counter()
            try:
                result = self.llm.embed(texts, task=task, dimensions=self.dimensions)
            except AllProvidersFailedError as exc:
                if "per day" in str(exc).lower() or "perday" in str(exc).lower():
                    # waiting minutes will not help: tell the caller to rerun after the reset
                    raise QuotaExhaustedError(
                        "daily embedding quota used up; what was embedded is cached, rerun later"
                    ) from exc
                if attempt == self.max_waits:
                    raise
                wait = self.retry_wait_s * (attempt + 1)
                self._progress(
                    f"rate limited or failing; waiting {wait:.0f}s (retry {attempt + 1})"
                )
                self._sleep(wait)
                continue
            if len(result.vectors) != len(texts):
                raise ValueError(f"asked for {len(texts)} embeddings, got {len(result.vectors)}")
            usage = getattr(self.llm, "usage", None)
            # latency of the successful attempt only: backoff waits after rate limits are excluded
            latency = usage[-1].latency_s if usage else time.perf_counter() - start
            return np.asarray(result.vectors, dtype=np.float32), latency
        raise AssertionError("unreachable")
