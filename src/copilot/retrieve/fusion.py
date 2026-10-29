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

    TODO(me): implement this.
    Hints:
      * Use only the RANKS, never the raw scores: a BM25 score and a cosine similarity live on
        different scales and cannot be added.
      * A dict id -> score plus one pass over each ranking (enumerate(..., start=1)) is enough.
      * Python's sorted() is stable, so sorting a dict's items by -score already keeps the
        first-seen order for ties (dicts remember insertion order).
      * Check your work with tests/test_retrieve_fusion_metrics.py (the xfail tests become
        passing), then set IMPL = "mine" in evals/run_retrieval.py and compare the numbers.
    """
    raise NotImplementedError("TODO(me): implement RRF (see the formula in the docstring)")
