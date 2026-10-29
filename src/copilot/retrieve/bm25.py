"""BM25 (rank-bm25) over the chunks, one index per language."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from copilot.retrieve.chunks import Corpus, Filters, chunk_text
from copilot.retrieve.tokenize import tokenize


class BM25Index:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._models = {
            lang: BM25Okapi([tokenize(chunk_text(u), lang) for u in chunks])
            for lang, chunks in corpus.by_language.items()
        }

    def search(self, query: str, filters: Filters, k: int = 50) -> list[str]:
        """Chunk ids ranked by BM25 score, best first. Chunks sharing no token are not returned."""
        return [chunk_id for chunk_id, _ in self.search_scored(query, filters, k)]

    def search_scored(self, query: str, filters: Filters, k: int = 50) -> list[tuple[str, float]]:
        """Like `search`, with each chunk's BM25 score (not comparable across different queries)."""
        scores = np.asarray(
            self._models[filters.language].get_scores(tokenize(query, filters.language))
        )
        scores = np.where(self.corpus.mask(filters), scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:k]
        chunks = self.corpus.by_language[filters.language]
        return [(chunks[i].id, float(scores[i])) for i in order if scores[i] > 0]
