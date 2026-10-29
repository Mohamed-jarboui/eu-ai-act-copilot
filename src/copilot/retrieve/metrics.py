"""Retrieval metrics against the gold set (rule and format: evals/README.md)."""

from __future__ import annotations


def is_hit(retrieved_id: str, gold_id: str) -> bool:
    """The README rule: equal to the gold id, or below it (the gold id followed by ':')."""
    return retrieved_id == gold_id or retrieved_id.startswith(gold_id + ":")


def recall_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    """Share of the gold ids that are found in the first k retrieved ids.

    Formula, for one question:

        recall@k = |{ g in gold : some r in ranked[:k] has is_hit(r, g) }| / |gold|

    A gold id counts once however many retrieved ids hit it (several paragraphs of Article 5 are
    one hit for "aiact:art:5"); a gold id that nothing in the top k hits counts as missed.
    The benchmark value is the mean over questions (the caller does that). `gold` is never empty
    here: out-of-scope questions are skipped.
    """
    top = ranked[:k]
    found = sum(1 for g in gold if any(is_hit(r, g) for r in top))
    return found / len(gold)


def reciprocal_rank(ranked: list[str], gold: list[str]) -> float:
    """Reciprocal rank of the first hit, for one question. The mean over questions is the MRR.

    Formula:

        RR  = 1 / rank_of_first_hit        (rank starts at 1: a hit in first place gives 1.0)
        RR  = 0                            if nothing in `ranked` hits any gold id
        MRR = mean of RR over the questions

    A "hit" is any retrieved id r with is_hit(r, g) for at least one g in gold.
    """
    for rank, r in enumerate(ranked, start=1):
        if any(is_hit(r, g) for g in gold):
            return 1.0 / rank
    return 0.0
