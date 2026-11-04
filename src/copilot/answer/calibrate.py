"""`python -m copilot.answer.calibrate`: the retrieval scores behind the refusal decision.

For every gold question this prints the best BM25 score (and per query token) and, where the
question embedding and that language's vectors are in the disk cache, the best cosine similarity.
It never calls an API, except with `--embed`, which fetches the question embeddings that are
missing (one call each). Use it to choose a refusal threshold, and say how many questions the
choice rests on: the gold set has only a few out-of-scope questions.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.cache import DiskCache
from copilot.retrieve.chunks import Corpus, Filters
from copilot.retrieve.embed import Embedder
from copilot.retrieve.tokenize import tokenize
from copilot.retrieve.vector import VectorIndex

GOLD = Path("evals/retrieval_v0.jsonl")
EXTRA_OUT_OF_SCOPE = Path("evals/refusal_v0.jsonl")  # more out-of-scope questions (English)


def questions() -> list[dict]:
    rows = [json.loads(line) for line in GOLD.read_text("utf-8").splitlines() if line.strip()]
    if EXTRA_OUT_OF_SCOPE.exists():
        rows += [
            json.loads(line)
            for line in EXTRA_OUT_OF_SCOPE.read_text("utf-8").splitlines()
            if line.strip()
        ]
    return rows


def fetch_missing_query_embeddings() -> None:
    """One API call per question whose embedding is not cached yet (needs GEMINI_API_KEY)."""
    from copilot.llm.gemini import GeminiProvider
    from copilot.llm.router import FallbackLLM
    from copilot.retrieve.embed import QuotaExhaustedError

    embedder = Embedder(
        FallbackLLM([GeminiProvider()], max_attempts=2, backoff_base_s=2.0),
        DiskCache(),
        dimensions=768,
        retry_wait_s=0,
        max_waits=0,
    )
    for q in questions():
        try:
            embedder.embed([q["question"]], "query")
        except QuotaExhaustedError:
            print("daily embedding quota used up; the rest stays unembedded")
            return
        except Exception as exc:  # keep going: one failure should not hide the others
            print(f"{q['id']}: {type(exc).__name__}")
    print("query embeddings fetched")


def collect() -> list[dict]:
    corpus = Corpus.load()
    bm25 = BM25Index(corpus)
    embedder = Embedder(None, DiskCache(), dimensions=768)
    vectors = VectorIndex.from_cache(corpus, embedder)
    rows = []
    for q in questions():
        filters = Filters(language=q["lang"])
        hits = bm25.search_scored(q["question"], filters, 1)
        tokens = len(tokenize(q["question"], q["lang"]))
        cosine = None
        query_vector = embedder.cached([q["question"]], "query")
        if query_vector is not None and vectors.has(q["lang"]):
            cosine = vectors.search_scored(query_vector[0], filters, 1)[0][1]
        top = hits[0][1] if hits else 0.0
        rows.append({**q, "tokens": tokens, "top_bm25": top, "per_token": top / max(tokens, 1),
                     "cosine": cosine})  # fmt: skip
    return rows


MIN_OUT_OF_SCOPE = 5  # fewer negatives than this in a language and no threshold is proposed


def choose_threshold(
    inside: list[float], outside: list[float], min_outside: int = MIN_OUT_OF_SCOPE
) -> float | None:
    """Propose a cosine threshold from measured scores, or None when the data does not support one.

    `inside` holds the best cosine of in-scope questions (they must NOT be refused), `outside`
    that of out-of-scope questions (they should be). Return None when there are no in-scope
    scores, fewer than `min_outside` out-of-scope ones, or when the groups overlap (some
    out-of-scope score >= the lowest in-scope score). Otherwise return the midpoint of the gap
    between the highest out-of-scope score and the lowest in-scope one, rounded to 3 decimals.
    """
    # TODO(me): implement the rule in the docstring.
    # Hint: three early `return None` checks, then the gap is (max(outside), min(inside)).
    # Placeholder: never propose a threshold, so nothing changes until the rule is written.
    return None


def by_language(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Known cosines per language, split into in-scope and out-of-scope."""
    groups: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r["cosine"] is None:
            continue
        group = groups.setdefault(r["lang"], {"inside": [], "outside": []})
        group["outside" if r["scope"] == "none" else "inside"].append(r["cosine"])
    return groups


def report_by_language(rows: list[dict]) -> None:
    from copilot.answer.refusal import COSINE_MIN

    print("\nper language (best cosine):")
    for lang, g in sorted(by_language(rows).items()):
        inside, outside = g["inside"], g["outside"]
        parts = [f"{lang}  in-scope n={len(inside)}"]
        if inside:
            parts.append(f"{min(inside):.3f}-{max(inside):.3f}")
        parts.append(f"| out-of-scope n={len(outside)}")
        if outside:
            parts.append(f"{min(outside):.3f}-{max(outside):.3f}")
        proposed = choose_threshold(inside, outside)
        parts.append("| proposed: " + ("none" if proposed is None else f"{proposed:.3f}"))
        current = COSINE_MIN.get(lang)
        if current is not None:
            wrong = sum(c < current for c in inside)
            caught = sum(c < current for c in outside)
            parts.append(f"| current {current}: refuses {wrong}/{len(inside)} in-scope, "
                         f"{caught}/{len(outside)} out-of-scope")  # fmt: skip
        print(" ".join(parts))


def main() -> None:
    if "--embed" in sys.argv:
        fetch_missing_query_embeddings()
    rows = collect()
    columns = ("id", "lang", "scope", "tokens", "top_bm25", "per_tok", "cosine")
    print(" ".join(f"{c:>8}" for c in columns))
    for r in rows:
        cos = "" if r["cosine"] is None else f"{r['cosine']:.3f}"
        values = (r["id"], r["lang"], r["scope"], r["tokens"], f"{r['top_bm25']:.2f}",
                  f"{r['per_token']:.2f}", cos)  # fmt: skip
        print(" ".join(f"{v!s:>8}" for v in values))
    inside = [r for r in rows if r["scope"] != "none"]
    outside = [r for r in rows if r["scope"] == "none"]
    print(f"\nin-scope questions: {len(inside)}, out-of-scope: {len(outside)}")
    for label, group in (("in-scope", inside), ("out-of-scope", outside)):
        bm25 = [r["per_token"] for r in group]
        cos = [r["cosine"] for r in group if r["cosine"] is not None]
        print(f"{label:13} BM25/token: min {min(bm25):.2f} median {statistics.median(bm25):.2f} "
              f"max {max(bm25):.2f} | cosine known for {len(cos)} of {len(group)}"
              + (f": min {min(cos):.3f} max {max(cos):.3f}" if cos else ""))  # fmt: skip
    report_by_language(rows)


if __name__ == "__main__":
    main()
