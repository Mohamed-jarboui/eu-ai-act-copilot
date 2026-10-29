# EU AI Act Copilot

Ask questions about the EU AI Act and GDPR in English, French or German and get answers grounded in the official texts, with article-level citations. An agent can also run compliance checks for a described AI system.

**Status:** in progress. Design notes: [docs/design.md](docs/design.md).

## How to run

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is installed by uv if missing.

```bash
make install            # uv sync
cp .env.example .env    # then add GEMINI_API_KEY and/or GROQ_API_KEY
make lint               # ruff check + format check
make test               # pytest (no network, fake providers)
make ingest             # parse data/raw into data/processed/corpus.jsonl (downloads missing files)
make eval-retrieval     # recall@k / MRR for BM25, vector, hybrid, hybrid+rerank (needs both API keys)
```

Corpus format, sources, licence and limitations: [docs/data-card.md](docs/data-card.md).

To run a script with your keys loaded: `uv run --env-file .env python your_script.py`.
