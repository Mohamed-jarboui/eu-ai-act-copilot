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

Result files are named after `--tag`. As of 2026-10-29 only `results/retrieval_bm25_all.*` exists: English chunks are embedded and cached, French and German are not, and the vector, hybrid and rerank numbers are still to be produced.
