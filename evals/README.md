# Evaluation sets

## retrieval_v0.jsonl

40 questions for measuring retrieval (recall@k, MRR). One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | `q01` … `q40` |
| `lang` | Question language: `en` (14), `fr` (13), `de` (13) |
| `scope` | `aiact` (23), `gdpr` (10), `both` (4), `none` (3, out of scope) |
| `question` | Written in plain words, not copied from the article text |
| `gold` | Corpus unit ids that answer the question, at article or annex level |

**Matching rule:** a retrieved unit is a hit for a gold id if its id equals it or starts with it followed by `:` (so `aiact:art:5:1:f` counts for `aiact:art:5`). Recitals do not count as hits.

**Out of scope:** the 3 `none` questions have an empty `gold` list. They are skipped for recall and MRR and are kept for testing refusals later.

Every gold id was checked against `data/processed/corpus.jsonl`. When a label looks wrong, fix it here in a separate commit with the reason in the commit message.

## Running the retrieval evaluation

`make eval-retrieval` (needs `GEMINI_API_KEY` and `GROQ_API_KEY` in `.env`) runs `evals/run_retrieval.py` and writes `results/retrieval.json` and `results/retrieval.md`. Document embeddings, query embeddings and rerank scores are cached in `data/cache/retrieval.sqlite`, so a rerun costs no API calls.

- **Scoring:** the 37 questions with a non-empty `gold` list. Each is searched in its own language over the chunks of both regulations (no filter on `scope`). Recitals are not indexed.
- **Definitions:** the corpus splits the definitions article (AI Act Art. 3, GDPR Art. 4) into `definition` units with ids like `aiact:def:1`, which do not start with `aiact:art:3:`. For matching, a `definition` unit counts as its parent article, so `aiact:def:1` is a hit for gold `aiact:art:3` (q15). Every other unit follows the rule above unchanged.
- **Metrics:** recall@k is the share of a question's gold ids hit within the first k retrieved chunks; MRR is the mean of 1/rank of the first hit (0 if none in the 50 retrieved). Several chunks of one article each take a slot in the list.
- **Cost:** token and character counts come from the run. Dollars are shown only if `evals/prices.json` has a price for the rerank model, in this format, with the source and date of the price:

```json
{"openai/gpt-oss-20b": {"input_per_1m_tokens_usd": 0.0, "output_per_1m_tokens_usd": 0.0,
                        "source": "URL", "date": "YYYY-MM-DD"}}
```

### Finishing the vector results (Gemini free-tier limit)

Gemini's free tier allows about 1000 embedding texts per day (HTTP 429 `...PerDay...`), and the three languages need about 3300 chunk embeddings plus one per question. The embedder fails fast on that error and everything embedded is cached on disk, so the evaluation can simply be rerun each day until the cache is full:

```bash
uv run --env-file .env python evals/run_retrieval.py --languages en --tag en    # English, all 4 methods
uv run --env-file .env python evals/run_retrieval.py                            # all languages -> results/retrieval.json
uv run --env-file .env python evals/run_retrieval.py --methods bm25 --tag bm25_all   # no API calls
```

Result files are named after `--tag`. `results/retrieval_bm25_all.*` is the earlier BM25-only run (2026-10-29).

**Daily rerun on the maintainer's machine (Windows):** a scheduled task `EuAiActCopilot-RetrievalEval` ran `scripts/daily_retrieval_eval.ps1` every day at 08:10 (Gemini's quota resets around 08:00 Tunisian time), resuming from the disk cache each time. The full run completed on 2026-11-01 and the script then removed the task. If the cache is ever lost, the same script can be scheduled again.

### Results (run 2026-11-01, `results/retrieval.json` and `results/retrieval.md`)

37 scored questions (13 EN, 12 FR, 12 DE), each searched in its own language over 1092 chunks. Rerunning from the cache on 2026-11-04 made no API calls and gave identical quality numbers.

| Method | recall@5 | recall@10 | MRR | p50 latency |
|---|---|---|---|---|
| bm25 | 0.694 | 0.797 | 0.652 | 1 ms |
| vector | 0.896 | 0.973 | 0.864 | 362 ms |
| hybrid | 0.878 | 0.946 | 0.798 | 365 ms |
| hybrid_rerank | 0.946 | 0.987 | 0.953 | 1395 ms |

- **Hybrid + rerank wins** on every overall metric: the first gold hit is at rank 1 for 34 of 37 questions (vector 29, hybrid 26, BM25 22). It costs one rerank call and about 1 s more at p50.
- **Vector beats plain hybrid.** Equal-weight RRF lets a poor BM25 ranking pull the right article down: hybrid's first hit is worse than vector's for 11 questions and better for 6. The gap is largest in German (MRR: vector 0.958, hybrid 0.722, BM25 0.578). In French, where BM25 is strong (0.772), hybrid beats vector (0.823 against 0.718).
- **Vectors fix BM25's vocabulary misses:** BM25 has 7 questions with no gold hit in the top 10, the other three methods have none.
- Per-language cells rest on 12 or 13 questions, and per-scope "both" cells on 4. Treat differences there as suggestive.
- The `rerank USD` figure in `results/retrieval.md` used the `openai/gpt-oss-20b` price from an `evals/prices.json` (Groq, 2026-10-31) that is not on `main` yet, so a rerun on `main` prints `n/a` there.

## refusal_v0.jsonl

12 English out-of-scope questions for calibrating the refusal threshold: 6 `unrelated` (r01-r06) and 6 `adjacent` (r07-r12, about other laws or ML practice). `make calibrate` uses them together with the 3 `none` questions in `retrieval_v0.jsonl`. Calibration as of 2026-11-04 (all 52 questions embedded): English out-of-scope scores 0.529 to 0.708 overlap English in-scope ones (0.699 to 0.814), so the threshold stays provisional. French and German have only one out-of-scope question each, so more are needed there.

## routes_v0.jsonl

77 questions for the agent's router: the 40 retrieval questions, the 12 extra out-of-scope ones (`refusal_v0.jsonl`), and 22 new ones (8 `lookup`, 8 `compliance_check`, 6 `compare`, in EN, FR and DE), plus 3 deliberately vague questions with `route: null` and `expect_clarify: true`. Labels were drafted by the assistant and are to be reviewed by the owner. Four questions that describe the asker's own situation and ask what applies (q12, q13, q26, q39) are marked `ambiguous`; `make eval-router` reports accuracy with and without them, and writes `results/router.json` and `results/router.md`.
