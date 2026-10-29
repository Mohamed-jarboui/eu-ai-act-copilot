"""`python -m copilot.answer.calibrate`: the retrieval scores behind the refusal decision.

For every gold question this prints the best BM25 score (and per query token) and, where the
question embedding and that language's vectors are in the disk cache, the best cosine similarity.
It never calls an API. Use it to choose a refusal threshold, and say how many questions the
choice rests on: the gold set has only a few out-of-scope questions.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.cache import DiskCache
from copilot.retrieve.chunks import Corpus, Filters
from copilot.retrieve.embed import Embedder
from copilot.retrieve.tokenize import tokenize
from copilot.retrieve.vector import VectorIndex

GOLD = Path("evals/retrieval_v0.jsonl")


def collect() -> list[dict]:
    corpus = Corpus.load()
    bm25 = BM25Index(corpus)
    embedder = Embedder(None, DiskCache(), dimensions=768)
    vectors = VectorIndex.from_cache(corpus, embedder)
    rows = []
    for line in GOLD.read_text("utf-8").splitlines():
        q = json.loads(line)
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


def main() -> None:
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


if __name__ == "__main__":
    main()
