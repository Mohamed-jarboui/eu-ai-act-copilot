"""The four retrieval methods behind one interface, each reporting what it cost."""

from __future__ import annotations

import time
from dataclasses import dataclass

from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.chunks import Corpus, Filters, chunk_text
from copilot.retrieve.embed import Embedder
from copilot.retrieve.rerank import Reranker
from copilot.retrieve.vector import VectorIndex

METHODS = ("bm25", "vector", "hybrid", "hybrid_rerank")
DEPTH = 50  # candidates taken from each ranker before fusion


@dataclass
class SearchResult:
    ids: list[str]
    local_s: float = 0.0  # time spent in local code (BM25, matrix product, fusion)
    api_s: float = 0.0  # recorded latency of the API calls this method needs (embed, rerank)
    embed_calls: int = 0
    embed_chars: int = 0
    rerank_calls: int = 0
    rerank_input_tokens: int | None = 0
    rerank_output_tokens: int | None = 0
    rerank_failed: bool = False


class Retriever:
    def __init__(
        self,
        corpus: Corpus,
        bm25: BM25Index,
        vectors: VectorIndex,
        embedder: Embedder,
        fuse,
        reranker: Reranker | None = None,
    ) -> None:
        self.corpus, self.bm25, self.vectors, self.embedder = corpus, bm25, vectors, embedder
        self.fuse, self.reranker = fuse, reranker
        # chunk ids repeat across languages, so the language is part of the key
        self.by_id = {
            (u.language, u.id): u for chunks in corpus.by_language.values() for u in chunks
        }

    def search(self, method: str, query: str, filters: Filters, k: int = DEPTH) -> SearchResult:
        if method == "bm25":
            t = time.perf_counter()
            ids = self.bm25.search(query, filters, k)
            return SearchResult(ids, local_s=time.perf_counter() - t)
        if method == "vector":
            return self._vector(query, filters, k)
        if method not in ("hybrid", "hybrid_rerank"):
            raise ValueError(f"unknown method {method!r}")
        t = time.perf_counter()
        lexical = self.bm25.search(query, filters, DEPTH)
        lexical_s = time.perf_counter() - t
        dense = self._vector(query, filters, DEPTH)
        t = time.perf_counter()
        fused = [doc_id for doc_id, _ in self.fuse([lexical, dense.ids])]
        result = SearchResult(
            fused[:k],
            local_s=lexical_s
            + dense.local_s
            + (time.perf_counter() - t),  # run one after the other
            api_s=dense.api_s,
            embed_calls=dense.embed_calls,
            embed_chars=dense.embed_chars,
        )
        if method == "hybrid_rerank":
            self._rerank(query, filters.language, fused, result, k)
        return result

    def _vector(self, query: str, filters: Filters, k: int) -> SearchResult:
        vectors, latencies = self.embedder.embed([query], "query")
        t = time.perf_counter()
        ids = self.vectors.search(vectors[0], filters, k)
        return SearchResult(
            ids,
            local_s=time.perf_counter() - t,
            api_s=latencies[0],  # recorded latency of the original query-embedding call
            embed_calls=1,
            embed_chars=len(query),
        )

    def _rerank(
        self, query: str, language: str, fused: list[str], result: SearchResult, k: int
    ) -> None:
        if self.reranker is None:
            raise RuntimeError("hybrid_rerank needs a reranker")
        candidates = [(i, chunk_text(self.by_id[(language, i)])) for i in fused]
        reranked = self.reranker.rerank(query, candidates)
        result.ids = reranked.ids[:k]
        result.api_s += reranked.api_latency_s
        result.rerank_calls = 0 if reranked.failed else 1
        result.rerank_input_tokens = reranked.input_tokens
        result.rerank_output_tokens = reranked.output_tokens
        result.rerank_failed = reranked.failed
