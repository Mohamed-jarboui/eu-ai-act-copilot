.PHONY: install lint test ingest eval-retrieval

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

ingest:
	uv run python -m copilot.ingest

eval-retrieval:
	uv run --env-file .env python evals/run_retrieval.py
