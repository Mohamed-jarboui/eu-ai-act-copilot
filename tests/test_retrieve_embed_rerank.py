"""Embedding cache, vector index, reranker and the four methods, with fakes (no network)."""

import numpy as np
import pytest

from copilot.llm.base import AllProvidersFailedError, Embedding, Generation, Usage
from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.cache import DiskCache
from copilot.retrieve.chunks import Corpus, Filters
from copilot.retrieve.embed import Embedder, EmbedStats, QuotaExhaustedError
from copilot.retrieve.impl import get_impl
from copilot.retrieve.rerank import Reranker, build_prompt, parse_scores
from copilot.retrieve.search import Retriever
from copilot.retrieve.vector import VectorIndex


class FakeEmbedLLM:
    """Embeds a text as a small vector derived from its words; can fail the first calls."""

    def __init__(self, fail_first: int = 0) -> None:
        self.calls: list[tuple[list[str], str, int | None]] = []
        self.fail_first = fail_first

    def embed(self, texts, task="document", dimensions=None):
        self.calls.append((list(texts), task, dimensions))
        if self.fail_first:
            self.fail_first -= 1
            raise AllProvidersFailedError("rate limited")
        words = ["chatbot", "prohibited", "breach"]
        vectors = [[float(w in t.lower()) + 0.01 for w in words] for t in texts]
        return Embedding(vectors=vectors, model="fake")


@pytest.fixture
def cache():
    db = DiskCache(":memory:")  # type: ignore[arg-type]
    yield db
    db.close()


# --- embedding cache -----------------------------------------------------------------------


def test_nothing_is_embedded_twice(cache):
    llm = FakeEmbedLLM()
    embedder = Embedder(llm, cache, dimensions=3)
    stats = EmbedStats()
    first, _ = embedder.embed(["a chatbot", "a breach"], "document", stats)
    second, _ = embedder.embed(["a chatbot", "a breach"], "document", stats)
    assert len(llm.calls) == 1
    assert np.allclose(first, second)
    assert (stats.texts, stats.cache_hits, stats.calls) == (4, 2, 1)


def test_task_is_part_of_the_cache_key(cache):
    llm = FakeEmbedLLM()
    embedder = Embedder(llm, cache, dimensions=3)
    embedder.embed(["a chatbot"], "document")
    embedder.embed(["a chatbot"], "query")
    assert [c[1] for c in llm.calls] == ["document", "query"]


def test_batches_and_unit_length_vectors(cache):
    llm = FakeEmbedLLM()
    embedder = Embedder(llm, cache, batch_size=2, dimensions=3)
    vectors, latencies = embedder.embed(["chatbot", "breach", "prohibited"], "document")
    assert [len(c[0]) for c in llm.calls] == [2, 1]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert len(latencies) == 3


def test_waits_and_retries_when_rate_limited(cache):
    llm, waits = FakeEmbedLLM(fail_first=2), []
    embedder = Embedder(llm, cache, dimensions=3, retry_wait_s=5, sleep=waits.append)
    embedder.embed(["chatbot"], "document")
    assert waits == [5, 10]  # linear backoff, and the third attempt succeeded


def test_gives_up_after_max_waits(cache):
    embedder = Embedder(
        FakeEmbedLLM(fail_first=99), cache, dimensions=3, max_waits=2, sleep=lambda _s: None
    )
    with pytest.raises(AllProvidersFailedError):
        embedder.embed(["chatbot"], "document")


# --- vector index and the four methods ---------------------------------------------------------


@pytest.fixture
def retriever(make_unit, cache):
    corpus = Corpus(
        [
            make_unit("aiact:art:5:1", "prohibited practices", article="5"),
            make_unit("aiact:art:50:1", "chatbot must say it is a machine", article="50"),
            make_unit("gdpr:art:33:1", "breach notification", regulation="gdpr", article="33"),
        ]
    )
    embedder = Embedder(FakeEmbedLLM(), cache, dimensions=3)
    vectors = VectorIndex.build(corpus, embedder)
    rerank_llm = FakeRerankLLM("[0, 3, 1]")
    reranker = Reranker(rerank_llm, cache, model_label="fake", min_interval_s=0)
    retriever = Retriever(
        corpus, BM25Index(corpus), vectors, embedder, get_impl().rrf_fuse, reranker
    )
    retriever.rerank_llm = rerank_llm
    return retriever


def test_vector_search_finds_the_closest_chunk(retriever):
    result = retriever.search("vector", "chatbot", Filters(language="en"))
    assert result.ids[0] == "aiact:art:50:1"
    assert result.embed_calls == 1 and result.embed_chars == len("chatbot")


def test_vector_search_applies_the_regulation_filter(retriever):
    result = retriever.search("vector", "chatbot", Filters(language="en", regulation="gdpr"))
    assert result.ids == ["gdpr:art:33:1"]


def test_type_filter_excludes_other_types(retriever):
    result = retriever.search(
        "bm25", "chatbot", Filters(language="en", types=frozenset({"article"}))
    )
    assert result.ids == []


def test_hybrid_fuses_both_rankers(retriever):
    result = retriever.search("hybrid", "chatbot breach", Filters(language="en"))
    assert set(result.ids) == {"aiact:art:50:1", "gdpr:art:33:1", "aiact:art:5:1"}
    assert result.embed_calls == 1 and result.rerank_calls == 0


def test_rerank_reorders_the_fused_list(retriever):
    fused = retriever.search("hybrid", "chatbot breach", Filters(language="en")).ids
    result = retriever.search("hybrid_rerank", "chatbot breach", Filters(language="en"))
    assert result.rerank_calls == 1 and not result.rerank_failed
    assert result.ids[0] == fused[1]  # the fake model scores the second candidate highest
    assert result.rerank_input_tokens == 100


def test_unknown_method(retriever):
    with pytest.raises(ValueError):
        retriever.search("magic", "x", Filters(language="en"))


# --- reranker ----------------------------------------------------------------------------------


class FakeRerankLLM:
    def __init__(self, answer: str) -> None:
        self.answer, self.calls, self.usage = answer, 0, []

    def generate(self, prompt, **kwargs):
        self.calls += 1
        self.usage.append(Usage("fake", "m", "generate", 100, 7, 0.25))
        return Generation(text=self.answer, model="m", input_tokens=100, output_tokens=7)


CANDIDATES = [("c1", "one"), ("c2", "two"), ("c3", "three"), ("c4", "four")]


@pytest.mark.parametrize(
    ("text", "n", "expected"),
    [
        ("[0, 3, 1]", 3, [0.0, 3.0, 1.0]),
        ("Here you go: [2,1]\nDone", 2, [2.0, 1.0]),
        ("[0, 3]", 3, None),  # wrong length
        ("no json", 2, None),
        ("[a, b]", 2, None),
    ],
)
def test_parse_scores(text, n, expected):
    assert parse_scores(text, n) == expected


def test_prompt_numbers_every_passage():
    prompt = build_prompt("q?", ["alpha", "beta"])
    assert "[1] alpha" in prompt and "[2] beta" in prompt and "exactly 2 integers" in prompt


def test_rerank_orders_by_score_and_keeps_ties_in_fused_order(cache):
    llm = FakeRerankLLM("[1, 3, 3]")
    reranker = Reranker(llm, cache, model_label="m", top_n=3, min_interval_s=0)
    result = reranker.rerank("q", CANDIDATES)
    assert result.ids == ["c2", "c3", "c1", "c4"]  # c4 was beyond top_n: kept at the end
    assert (result.api_latency_s, result.input_tokens, result.output_tokens) == (0.25, 100, 7)


def test_rerank_is_cached(cache):
    llm = FakeRerankLLM("[0, 1]")
    reranker = Reranker(llm, cache, model_label="m", top_n=2, min_interval_s=0)
    first = reranker.rerank("q", CANDIDATES)
    second = reranker.rerank("q", CANDIDATES)
    assert llm.calls == 1
    assert first.ids == second.ids == ["c2", "c1", "c3", "c4"]
    assert second.api_latency_s == 0.25  # the recorded latency of the first call


def test_unparsable_answer_keeps_the_original_order(cache):
    reranker = Reranker(FakeRerankLLM("sorry"), cache, model_label="m", min_interval_s=0)
    result = reranker.rerank("q", CANDIDATES)
    assert result.failed and result.ids == ["c1", "c2", "c3", "c4"]


def test_rerank_waits_between_uncached_calls(cache):
    waits, now = [], [0.0]
    reranker = Reranker(
        FakeRerankLLM("[1]"),
        cache,
        model_label="m",
        top_n=1,
        min_interval_s=2.0,
        sleep=waits.append,
        clock=lambda: now[0],
    )
    reranker.rerank("q1", CANDIDATES)
    now[0] = 0.5
    reranker.rerank("q2", CANDIDATES)
    assert waits == [pytest.approx(1.5)]


def test_daily_quota_fails_fast_without_waiting(cache):
    class DailyQuotaLLM:
        def embed(self, texts, task="document", dimensions=None):
            raise AllProvidersFailedError("429 ... Embed content requests per day limit exceeded")

    waits = []
    embedder = Embedder(DailyQuotaLLM(), cache, dimensions=3, sleep=waits.append)
    with pytest.raises(QuotaExhaustedError):
        embedder.embed(["chatbot"], "document")
    assert waits == []  # waiting minutes cannot help against a per-day limit
