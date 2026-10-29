"""A small in-memory vector index: unit-length vectors, brute-force cosine similarity (numpy)."""

from __future__ import annotations

import numpy as np

from copilot.retrieve.chunks import Corpus, Filters, chunk_text
from copilot.retrieve.embed import Embedder, EmbedStats


class VectorIndex:
    def __init__(self, corpus: Corpus, matrices: dict[str, np.ndarray]) -> None:
        self.corpus = corpus
        self.matrices = matrices  # language -> (n_chunks, dims), rows aligned with the corpus

    @classmethod
    def build(cls, corpus: Corpus, embedder: Embedder, stats: EmbedStats | None = None):
        matrices = {}
        for lang, chunks in corpus.by_language.items():
            matrices[lang], _ = embedder.embed([chunk_text(u) for u in chunks], "document", stats)
        return cls(corpus, matrices)

    @classmethod
    def from_cache(cls, corpus: Corpus, embedder: Embedder) -> VectorIndex:
        """Load the languages whose chunks are ALL in the disk cache; never calls the API."""
        matrices = {}
        for lang, chunks in corpus.by_language.items():
            cached = embedder.cached([chunk_text(u) for u in chunks], "document")
            if cached is not None:
                matrices[lang] = cached
        return cls(corpus, matrices)

    def has(self, language: str) -> bool:
        return language in self.matrices

    def search(self, query_vector: np.ndarray, filters: Filters, k: int = 50) -> list[str]:
        """Chunk ids ranked by cosine similarity to the (unit-length) query vector, best first."""
        return [chunk_id for chunk_id, _ in self.search_scored(query_vector, filters, k)]

    def search_scored(
        self, query_vector: np.ndarray, filters: Filters, k: int = 50
    ) -> list[tuple[str, float]]:
        """Like `search`, with each chunk's cosine similarity."""
        scores = self.matrices[filters.language] @ query_vector
        scores = np.where(self.corpus.mask(filters), scores, -np.inf)
        order = np.argsort(-scores, kind="stable")[:k]
        chunks = self.corpus.by_language[filters.language]
        return [(chunks[i].id, float(scores[i])) for i in order if np.isfinite(scores[i])]
