# Retrieval results

Gold set: `evals/retrieval_v0.jsonl`, 37 questions scored (3 out-of-scope skipped). Run 2026-11-01, implementation `mine`. Languages: en, fr, de. Each question is searched in its own language over 1092 chunks (both regulations, no recitals).

## Quality

| Method | recall@5 | recall@10 | MRR |
|---|---|---|---|
| bm25 | 0.694 | 0.797 | 0.652 |
| vector | 0.896 | 0.973 | 0.864 |
| hybrid | 0.878 | 0.946 | 0.798 |
| hybrid_rerank | 0.946 | 0.987 | 0.953 |

### By question language (n = number of questions)

| Method | de recall@10 / MRR (n=12) | en recall@10 / MRR (n=13) | fr recall@10 / MRR (n=12) |
|---|---|---|---|
| bm25 | 0.625 / 0.578 | 0.846 / 0.610 | 0.917 / 0.772 |
| vector | 1.000 / 0.958 | 0.962 / 0.910 | 0.958 / 0.718 |
| hybrid | 0.917 / 0.722 | 0.962 / 0.846 | 0.958 / 0.823 |
| hybrid_rerank | 0.958 / 0.958 | 1.000 / 1.000 | 1.000 / 0.896 |

## Latency and cost per query

Latency = local search time + recorded latency of the API calls the method needs (query embedding, rerank), added one after the other. Query embeddings and rerank scores are cached on disk; the cache stores the latency measured on the first call.

| Method | p50 ms | p95 ms | embed calls | embed chars | rerank in/out tok | rerank USD |
|---|---|---|---|---|---|---|
| bm25 | 1 | 2 | 0 | 0 | 0 / 0 | 0.000000 |
| vector | 362 | 1287 | 1 | 70 | 0 / 0 | 0.000000 |
| hybrid | 365 | 1288 | 1 | 70 | 0 / 0 | 0.000000 |
| hybrid_rerank | 1395 | 2313 | 1 | 70 | 1698 / 238 | 0.007356 |

## One-time index build (document embeddings)

3276 chunks (3256 distinct texts, 1745750 characters, 2 cut to 6000 characters) for the languages above. API calls made by this run: 5 (0 when the disk cache was already full). Gemini's free tier caps embedding at about 1000 texts per day, so the cache may have been filled over several runs. Gemini does not report embedding tokens, so no token or dollar figure is given.
