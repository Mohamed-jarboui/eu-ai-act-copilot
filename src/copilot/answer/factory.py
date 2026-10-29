"""Build the real pipeline from the corpus, the disk cache and the API keys in the environment."""

from __future__ import annotations

from copilot.answer.impl import IMPL, RETRIEVE_IMPL, get_impl
from copilot.answer.pipeline import AnswerPipeline
from copilot.answer.retrieval import AppRetriever
from copilot.llm.gemini import DEFAULT_MODELS as GEMINI_MODELS
from copilot.llm.gemini import GeminiProvider
from copilot.llm.groq import GroqProvider
from copilot.llm.router import FallbackLLM
from copilot.retrieve import impl as retrieve_impl
from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.cache import DiskCache
from copilot.retrieve.chunks import Corpus
from copilot.retrieve.embed import Embedder
from copilot.retrieve.vector import VectorIndex


def build_pipeline(impl_name: str = IMPL) -> AnswerPipeline:
    corpus = Corpus.load()
    # Query embeddings must fail fast (daily quota, network): the retriever then uses BM25 only.
    embed_llm = FallbackLLM([GeminiProvider()], max_attempts=2, backoff_base_s=1.0)
    embedder = Embedder(embed_llm, DiskCache(), retry_wait_s=0, max_waits=0)
    retriever = AppRetriever(
        corpus,
        BM25Index(corpus),
        VectorIndex.from_cache(corpus, embedder),
        embedder,
        retrieve_impl.get_impl(RETRIEVE_IMPL).rrf_fuse,
    )
    # Order matters for latency. The lite model answered every test call in 1-2 s; the larger
    # gemini-3.6-flash answered 503 "high demand" or took 7-22 s, and each failure costs a wait
    # before the fallback. So: lite first, the larger model second, Groq last (its free tier
    # allows only 8000 tokens per minute on the 120B model). Timeouts are short for the same reason.
    lite = GEMINI_MODELS["small"]
    gemini_lite = GeminiProvider(models={"small": lite, "large": lite}, timeout_s=30)
    gemini_large = GeminiProvider(
        thinking_level="MINIMAL", timeout_s=30
    )  # extracting, not reasoning
    gemini_check, groq_generation = GeminiProvider(), GroqProvider(timeout_s=30)
    groq_check = GroqProvider(reasoning_effort="low")  # the checker only needs a short verdict

    def generation_llm() -> FallbackLLM:  # one wrapper per request keeps usage records separate
        return FallbackLLM(
            [gemini_lite, gemini_large, groq_generation], max_attempts=2, backoff_base_s=3.0
        )

    def check_llm() -> FallbackLLM:
        return FallbackLLM([groq_check, gemini_check], max_attempts=3, backoff_base_s=3.0)

    return AnswerPipeline(retriever, generation_llm, check_llm, get_impl(impl_name))
