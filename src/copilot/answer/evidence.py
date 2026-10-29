"""What retrieval found, in the numbers the refusal decision needs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Evidence:
    """Signals about how well the best sources match the question.

    `top_cosine` is the cosine similarity of the best vector hit (between -1 and 1; None when the
    request ran without vectors). `top_bm25` is the best BM25 score, which depends on the query
    length and the corpus, so compare it to `query_tokens` and never across questions as such.
    """

    mode: str  # "hybrid" or "bm25"
    n_sources: int  # chunks handed to the model
    top_cosine: float | None
    top_bm25: float | None
    query_tokens: int  # tokens left in the question after stopword removal
    language: str = "en"  # cosine scales differ per language, so thresholds are per language
