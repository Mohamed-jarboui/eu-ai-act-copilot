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
