"""Retrieval for the app: hybrid (BM25 + vectors, fused with RRF) when possible, else BM25 alone."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from copilot.answer.evidence import Evidence
from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.chunks import Corpus, Filters
from copilot.retrieve.embed import Embedder
from copilot.retrieve.tokenize import tokenize
from copilot.retrieve.vector import VectorIndex

DEPTH = 50  # candidates taken from each ranker before fusion


@dataclass
class Retrieved:
    ids: list[str]  # chunk ids, best first (already cut to k)
    evidence: Evidence
    latency_s: float
    notes: list[str] = field(default_factory=list)  # why the app fell back to BM25, if it did


class AppRetriever:
    def __init__(
        self,
        corpus: Corpus,
        bm25: BM25Index,
        vectors: VectorIndex,
        embedder: Embedder,
        fuse,
    ) -> None:
        self.corpus, self.bm25, self.vectors, self.embedder, self.fuse = (
            corpus,
            bm25,
            vectors,
            embedder,
            fuse,
        )
        self.by_id = {
            (u.language, u.id): u for chunks in corpus.by_language.values() for u in chunks
        }

    def modes(self) -> dict[str, str]:
        """Per language, what a request will use: "hybrid" if its vectors are cached."""
        return {
            lang: "hybrid" if self.vectors.has(lang) else "bm25" for lang in self.corpus.by_language
        }

    def retrieve(
        self, query: str, language: str, k: int = 8, regulation: str | None = None
    ) -> Retrieved:
        start = time.perf_counter()
        filters = Filters(language=language, regulation=regulation)
        lexical = self.bm25.search_scored(query, filters, DEPTH)
        top_bm25 = lexical[0][1] if lexical else None
        n_tokens = len(tokenize(query, language))
        notes: list[str] = []

        dense: list[tuple[str, float]] = []
        if self.vectors.has(language):
            try:
                vectors, _ = self.embedder.embed([query], "query")
                dense = self.vectors.search_scored(vectors[0], filters, DEPTH)
            except (
                Exception
            ) as exc:  # quota spent, network down...: degrade, do not fail the request
                notes.append(f"query embedding unavailable ({type(exc).__name__}); used BM25 only")
        else:
            notes.append(f"no cached vectors for '{language}'; used BM25 only")

        if dense:
            ids = [i for i, _ in self.fuse([[i for i, _ in lexical], [i for i, _ in dense]])]
            mode, top_cosine = "hybrid", dense[0][1]
        else:
            ids = [i for i, _ in lexical]
            mode, top_cosine = "bm25", None
        ids = ids[:k]
        evidence = Evidence(mode, len(ids), top_cosine, top_bm25, n_tokens)
        return Retrieved(ids, evidence, time.perf_counter() - start, notes)
