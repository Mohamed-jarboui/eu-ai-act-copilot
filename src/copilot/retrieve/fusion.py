"""Reciprocal rank fusion: merge several ranked lists into one."""

from __future__ import annotations


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists (best first) with Reciprocal Rank Fusion.

    Formula, for each document d:

        score(d) = sum over every ranking r that contains d of   1 / (k + rank_r(d))

    where rank_r(d) is the 1-based position of d in ranking r (the first item has rank 1).
    A document missing from a ranking adds nothing for that ranking. k = 60 is the usual constant;
    a larger k flattens the difference between rank 1 and rank 10.

    Return (id, score) pairs sorted by score, highest first. When two documents have the same score,
    keep the one that appears first when scanning the rankings in the order given (first ranking
    first, top to bottom).

    Only the ranks are used, never the raw scores: a BM25 score and a cosine similarity live on
    different scales and cannot be added. A dict id -> score keeps insertion order, and sorted() is
    stable, so documents with equal scores keep their first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])
