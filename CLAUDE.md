# EU AI Act Copilot

RAG + agent system answering questions about the EU AI Act (Regulation (EU) 2024/1689) and GDPR (Regulation (EU) 2016/679) in English, French and German, with article-level citations. Design decisions live in `docs/design.md`: read it before changing architecture.

## Rules for all sessions

### Learning mode
- The owner is learning while building. Before coding, explain the plan in 5-8 bullets and wait for their OK.
- Leave 1-3 key pieces per task as `# TODO(me):` with a hint, for the owner to implement. Put a conservative placeholder next to it so tests still run.
- End each session with 3 questions that check the owner's understanding.

### Commits and PRs
- No AI attribution anywhere: no co-author lines, no "generated with" text, in commits or PR descriptions.
- Conventional commit style (`feat:`, `fix:`, `chore:`, `test:`, `ci:`, `docs:`).
- Work on a branch, never directly on `main`; open a PR into `main`.

### Numbers and claims
- Never invent metrics. Every number in docs must come from a script run, stated with its sample size.

### Secrets
- Secrets only in `.env` (git-ignored) or GitHub Secrets. Never read, print or commit `.env*` files; `.env.example` holds names only.

### Dependencies
- Keep dependencies light (the owner's local internet is slow). Prefer the standard library; justify every new package.

## Commands

```bash
make install   # uv sync
make lint      # ruff check + ruff format --check
make test      # pytest
```

## Layout

- `src/copilot/` - package code (`llm/` is the provider layer: Gemini -> Groq fallback, retries, usage records)
- `tests/` - pytest; no network, use fake providers
- `docs/design.md` - design notes
