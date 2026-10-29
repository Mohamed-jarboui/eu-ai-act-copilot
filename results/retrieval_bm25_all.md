# Retrieval results

Gold set: `evals/retrieval_v0.jsonl`, 37 questions scored (3 out-of-scope skipped). Run 2026-10-29, implementation `reference`. Languages: en, fr, de. Each question is searched in its own language over 1092 chunks (both regulations, no recitals).

## Quality

| Method | recall@5 | recall@10 | MRR |
|---|---|---|---|
| bm25 | 0.694 | 0.797 | 0.652 |

### By question language (n = number of questions)

| Method | de recall@10 / MRR (n=12) | en recall@10 / MRR (n=13) | fr recall@10 / MRR (n=12) |
|---|---|---|---|
| bm25 | 0.625 / 0.578 | 0.846 / 0.610 | 0.917 / 0.772 |

## Latency and cost per query

Latency = local search time + recorded latency of the API calls the method needs (query embedding, rerank), added one after the other. Query embeddings and rerank scores are cached on disk; the cache stores the latency measured on the first call.

| Method | p50 ms | p95 ms | embed calls | embed chars | rerank in/out tok | rerank USD |
|---|---|---|---|---|---|---|
| bm25 | 1 | 2 | 0 | 0 | 0 / 0 | 0.000000 |
