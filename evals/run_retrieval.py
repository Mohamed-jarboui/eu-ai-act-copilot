"""Retrieval evaluation: recall@5, recall@10 and MRR for BM25, vector, hybrid and hybrid+rerank.

Run with:  uv run --env-file .env python evals/run_retrieval.py     (or: make eval-retrieval)

Protocol (see evals/README.md for the gold set and the matching rule):
  * 37 of the 40 questions are scored; the 3 out-of-scope ones (scope "none") are skipped.
  * Each question is searched in its own language, over all chunks of both regulations
    (no oracle filter on the question's scope). Recitals are not indexed.
  * A retrieved chunk is matched through `anchor_id` (a definition counts as its article).
  * Lists are 50 deep; a first hit beyond rank 50 scores 0 for MRR.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from copilot.ingest.corpus import read_corpus
from copilot.ingest.download import LANGUAGES
from copilot.llm.gemini import GeminiProvider
from copilot.llm.groq import DEFAULT_MODELS as GROQ_MODELS
from copilot.llm.groq import GroqProvider
from copilot.llm.router import FallbackLLM
from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.cache import DiskCache
from copilot.retrieve.chunks import Corpus, Filters, anchor_id, chunk_text
from copilot.retrieve.embed import (
    DIMENSIONS,
    EMBED_MODEL,
    MAX_EMBED_CHARS,
    Embedder,
    EmbedStats,
)
from copilot.retrieve.impl import get_impl
from copilot.retrieve.metrics import is_hit
from copilot.retrieve.rerank import Reranker
from copilot.retrieve.search import DEPTH, METHODS, Retriever
from copilot.retrieve.vector import VectorIndex

IMPL = "mine"

GOLD = Path("evals/retrieval_v0.jsonl")
PRICES = Path("evals/prices.json")  # optional, see evals/README.md
RESULTS_JSON = Path("results/retrieval.json")
RESULTS_MD = Path("results/retrieval.md")
KS = (5, 10)
RERANK_MODEL = GROQ_MODELS["small"]


def load_questions(languages: list[str]) -> tuple[list[dict], int]:
    """(scored questions, number of out-of-scope questions skipped) for the chosen languages."""
    rows = [json.loads(line) for line in GOLD.read_text("utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["lang"] in languages]
    scored = [r for r in rows if r["scope"] != "none" and r["gold"]]
    return scored, len(rows) - len(scored)


def verify_models(methods: list[str]) -> None:
    missing = []
    if set(methods) - {"bm25"} and EMBED_MODEL not in GeminiProvider().available_models():
        missing.append(f"gemini: {EMBED_MODEL}")
    if "hybrid_rerank" in methods and RERANK_MODEL not in GroqProvider().available_models():
        missing.append(f"groq: {RERANK_MODEL}")
    if missing:
        raise SystemExit(f"model(s) not offered by this key: {', '.join(missing)}")


def first_hit_rank(anchors: list[str], gold: list[str]) -> int | None:
    for rank, anchor in enumerate(anchors, start=1):
        if any(is_hit(anchor, g) for g in gold):
            return rank
    return None


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def summarise(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        **{f"recall@{k}": round(mean(r[f"recall@{k}"] for r in rows), 4) for k in KS},
        "mrr": round(mean(r["rr"] for r in rows), 4),
    }


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def run_method(method: str, questions: list[dict], retriever: Retriever, impl) -> dict:
    rows, totals_ms, local_ms, api_ms = [], [], [], []
    usage = {
        "embed_calls": 0,
        "embed_chars": 0,
        "rerank_calls": 0,
        "rerank_input_tokens": 0,
        "rerank_output_tokens": 0,
        "rerank_failed": 0,
    }
    for q in questions:
        result = retriever.search(method, q["question"], Filters(language=q["lang"]))
        anchors = [anchor_id(retriever.by_id[(q["lang"], i)]) for i in result.ids]
        row = {
            "id": q["id"],
            "lang": q["lang"],
            "scope": q["scope"],
            "gold": q["gold"],
            "first_hit_rank": first_hit_rank(anchors, q["gold"]),
            "rr": impl.reciprocal_rank(anchors, q["gold"]),
            **{f"recall@{k}": impl.recall_at_k(anchors, q["gold"], k) for k in KS},
            "top5": anchors[:5],
        }
        rows.append(row)
        local_ms.append(result.local_s * 1000)
        api_ms.append(result.api_s * 1000)
        totals_ms.append((result.local_s + result.api_s) * 1000)
        usage["embed_calls"] += result.embed_calls
        usage["embed_chars"] += result.embed_chars
        usage["rerank_calls"] += result.rerank_calls
        usage["rerank_input_tokens"] += result.rerank_input_tokens or 0
        usage["rerank_output_tokens"] += result.rerank_output_tokens or 0
        usage["rerank_failed"] += int(result.rerank_failed)
    by_lang = {
        lang: summarise([r for r in rows if r["lang"] == lang])
        for lang in sorted({r["lang"] for r in rows})
    }
    by_scope = {
        s: summarise([r for r in rows if r["scope"] == s])
        for s in sorted({r["scope"] for r in rows})
    }
    return {
        "overall": summarise(rows),
        "by_language": by_lang,
        "by_scope": by_scope,
        "latency_ms": {
            "mean": round(mean(totals_ms), 2),
            "p50": round(pct(totals_ms, 50), 2),
            "p95": round(pct(totals_ms, 95), 2),
            "mean_local": round(mean(local_ms), 2),
            "mean_api_recorded": round(mean(api_ms), 2),
        },
        "usage_total": usage,
        "usage_per_query": {k: round(v / len(questions), 2) for k, v in usage.items()},
        "questions": rows,
    }


def rerank_cost_usd(usage: dict) -> float | None:
    """Dollars for the rerank calls, only if evals/prices.json has a sourced price for the model."""
    if not PRICES.exists():
        return None
    entry = json.loads(PRICES.read_text("utf-8")).get(RERANK_MODEL)
    if not entry:
        return None
    return round(
        usage["rerank_input_tokens"] / 1e6 * entry["input_per_1m_tokens_usd"]
        + usage["rerank_output_tokens"] / 1e6 * entry["output_per_1m_tokens_usd"],
        6,
    )


def markdown(report: dict) -> str:
    cfg, results = report["config"], report["methods"]
    lines = [
        "# Retrieval results",
        "",
        f"Gold set: `evals/retrieval_v0.jsonl`, {cfg['questions_scored']} questions scored "
        f"({cfg['questions_skipped']} out-of-scope skipped). Run {cfg['date']}, "
        f"implementation `{cfg['impl']}`. Languages: {', '.join(cfg['languages'])}. Each question "
        f"is searched in its own language over {max(cfg['chunks_per_language'].values())} chunks "
        f"(both regulations, no recitals).",
        "",
        "## Quality",
        "",
        "| Method | recall@5 | recall@10 | MRR |",
        "|---|---|---|---|",
    ]
    for m, r in results.items():
        o = r["overall"]
        lines.append(f"| {m} | {o['recall@5']:.3f} | {o['recall@10']:.3f} | {o['mrr']:.3f} |")
    first = next(iter(results.values()))
    langs = list(first["by_language"])
    lines += [
        "",
        "### By question language (n = number of questions)",
        "",
        "| Method | "
        + " | ".join(
            f"{lang} recall@10 / MRR (n={first['by_language'][lang]['n']})" for lang in langs
        )
        + " |",
        "|---|" + "---|" * len(langs),
    ]
    for m, r in results.items():
        cells = [
            f"{r['by_language'][lang]['recall@10']:.3f} / {r['by_language'][lang]['mrr']:.3f}"
            for lang in langs
        ]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Latency and cost per query",
        "",
        "Latency = local search time + recorded latency of the API calls the method needs "
        "(query embedding, rerank), added one after the other. Query embeddings and rerank "
        "scores are cached on disk; the cache stores the latency measured on the first call.",
        "",
        "| Method | p50 ms | p95 ms | embed calls | embed chars | rerank in/out tok | rerank USD |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, r in results.items():
        u, t = r["usage_per_query"], r["usage_total"]
        usd = rerank_cost_usd(t) if m == "hybrid_rerank" else 0.0
        usd_s = "n/a (no prices file)" if usd is None else f"{usd:.6f}"
        lines.append(
            f"| {m} | {r['latency_ms']['p50']:.0f} | {r['latency_ms']['p95']:.0f} | "
            f"{u['embed_calls']:.0f} | {u['embed_chars']:.0f} | "
            f"{u['rerank_input_tokens']:.0f} / {u['rerank_output_tokens']:.0f} | {usd_s} |"
        )
    if any(m != "bm25" for m in results):
        ib = report["index_build"]
        lines += [
            "",
            "## One-time index build (document embeddings)",
            "",
            f"{ib['chunks']} chunks ({ib['unique_texts']} distinct texts, {ib['chars']} "
            f"characters, {ib['truncated']} cut to {cfg['max_embed_chars']} characters) for the "
            f"languages above. API calls made by this run: {ib['api_calls_this_run']} (0 when the "
            f"disk cache was already full). Gemini's free tier caps embedding at about 1000 texts "
            f"per day, so the cache may have been filled over several runs. Gemini does not "
            f"report embedding tokens, so no token or dollar figure is given.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--tag", default="", help="suffix for the result files, e.g. 'en'")
    parser.add_argument("--skip-verify", action="store_true", help="do not check model names")
    args = parser.parse_args()
    impl = get_impl(IMPL)
    suffix = f"_{args.tag}" if args.tag else ""
    results_json = RESULTS_JSON.with_name(f"retrieval{suffix}.json")
    results_md = RESULTS_MD.with_name(f"retrieval{suffix}.md")
    if not args.skip_verify:
        verify_models(args.methods)

    questions, skipped = load_questions(args.languages)
    corpus = Corpus([u for u in read_corpus() if u.language in args.languages])
    cache = DiskCache()
    needs_vectors = bool(set(args.methods) - {"bm25"})
    needs_rerank = "hybrid_rerank" in args.methods
    embedder, vectors, reranker = None, None, None
    build_stats = EmbedStats()
    if needs_vectors:
        embed_llm = FallbackLLM([GeminiProvider()], max_attempts=5, backoff_base_s=2.0)
        embedder = Embedder(embed_llm, cache, progress=print)
        t0 = time.perf_counter()
        vectors = VectorIndex.build(corpus, embedder, build_stats)
        print(f"index ready in {time.perf_counter() - t0:.1f}s: {build_stats}")
    if needs_rerank:
        rerank_llm = FallbackLLM(
            [GroqProvider(reasoning_effort="low")], max_attempts=6, backoff_base_s=3.0
        )
        reranker = Reranker(rerank_llm, cache, model_label=RERANK_MODEL)
    retriever = Retriever(corpus, BM25Index(corpus), vectors, embedder, impl.rrf_fuse, reranker)

    methods = {}
    for method in args.methods:
        print(f"== {method}")
        methods[method] = run_method(method, questions, retriever, impl)
        print(methods[method]["overall"])

    texts = [chunk_text(u) for chunks in corpus.by_language.values() for u in chunks]
    distinct = set(texts)
    index_build = {  # derived from the corpus, so it does not depend on which run filled the cache
        "chunks": len(texts),
        "unique_texts": len(distinct),
        "chars": sum(min(len(t), MAX_EMBED_CHARS) for t in distinct),
        "truncated": sum(1 for t in distinct if len(t) > MAX_EMBED_CHARS),
        "api_calls_this_run": build_stats.calls,
    }
    report = {
        "config": {
            "date": date.today().isoformat(),
            "impl": IMPL,
            "languages": args.languages,
            "embed_model": EMBED_MODEL,
            "embed_dimensions": DIMENSIONS,
            "rerank_model": RERANK_MODEL,
            "rrf_k": 60,
            "depth": DEPTH,
            "questions_scored": len(questions),
            "questions_skipped": skipped,
            "chunks_per_language": {lang: len(c) for lang, c in corpus.by_language.items()},
            "max_embed_chars": MAX_EMBED_CHARS,
        },
        "index_build": index_build,
        "methods": methods,
    }
    results_json.parent.mkdir(exist_ok=True)
    results_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    results_md.write_text(markdown(report), "utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
