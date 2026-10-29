"""Reference versions of the TODO(me) functions, so the evaluation can run before they are done.

Spoiler: these are the answers to fusion.py and metrics.py. Try your own first. The evaluation
runner records which implementation produced a result, and yours must reproduce the same numbers.
"""

from __future__ import annotations

from copilot.retrieve.metrics import is_hit


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])  # sorted() is stable: ties keep order


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    top = ranked[:k]
    found = sum(1 for g in gold if any(is_hit(r, g) for r in top))
    return found / len(gold)


def reciprocal_rank(ranked: list[str], gold: list[str]) -> float:
    for rank, r in enumerate(ranked, start=1):
        if any(is_hit(r, g) for g in gold):
            return 1.0 / rank
    return 0.0
