"""Optional reranker: a small LLM scores the top candidates for relevance to the question."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from copilot.retrieve.cache import DiskCache, digest

PROMPT_VERSION = "v1"
SYSTEM = "You judge how relevant legal passages are to a question. You answer only with JSON."


@dataclass
class RerankResult:
    ids: list[str]  # candidate ids, best first
    api_latency_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    failed: bool = False  # the model's answer could not be parsed: original order kept


def build_prompt(query: str, passages: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return (
        f"Question: {query}\n\nPassages:\n{numbered}\n\n"
        f"Give each passage an integer relevance score from 0 (irrelevant) to 3 (directly answers "
        f"the question). Reply with ONLY a JSON array of exactly {len(passages)} integers, in the "
        f"order of the passages, for example [0, 3, 1]."
    )


def parse_scores(text: str, n: int) -> list[int] | None:
    match = re.search(r"\[[^\[\]]*\]", text)
    if not match:
        return None
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if len(scores) != n or not all(isinstance(s, int | float) for s in scores):
        return None
    return [float(s) for s in scores]


class Reranker:
    def __init__(
        self,
        llm,
        cache: DiskCache,
        *,
        model_label: str,
        top_n: int = 20,
        passage_chars: int = 400,
        min_interval_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm, self.cache, self.model_label = llm, cache, model_label
        self.top_n, self.passage_chars, self.min_interval_s = top_n, passage_chars, min_interval_s
        self._sleep, self._clock, self._last_call = sleep, clock, None

    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> RerankResult:
        """`candidates` is (id, text) best-first; only the first `top_n` are scored."""
        head = candidates[: self.top_n]
        ids = [cid for cid, _ in head]
        key = digest(self.model_label, PROMPT_VERSION, self.passage_chars, query, "|".join(ids))
        payload = self.cache.get_rerank(key)
        if payload is None:
            payload = self._ask(query, [t[: self.passage_chars] for _, t in head], len(head))
            if payload is None:
                return RerankResult(ids=[cid for cid, _ in candidates], failed=True)
            self.cache.put_rerank(key, payload)
        scores = payload["scores"]
        order = sorted(range(len(head)), key=lambda i: (-scores[i], i))  # ties keep fused order
        return RerankResult(
            ids=[ids[i] for i in order] + [cid for cid, _ in candidates[len(head) :]],
            api_latency_s=payload["api_latency_s"],
            input_tokens=payload["input_tokens"],
            output_tokens=payload["output_tokens"],
        )

    def _ask(self, query: str, passages: list[str], n: int) -> dict | None:
        if self._last_call is not None:
            wait = self.min_interval_s - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        result = self.llm.generate(
            build_prompt(query, passages), system=SYSTEM, tier="small", max_output_tokens=1000
        )
        self._last_call = self._clock()
        scores = parse_scores(result.text, n)
        if scores is None:
            return None
        usage = self.llm.usage[-1]  # the successful attempt: waits caused by rate limits excluded
        return {
            "scores": scores,
            "api_latency_s": usage.latency_s,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
