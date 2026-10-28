.PHONY: install lint test ingest

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

ingest:
	uv run python -m copilot.ingest
