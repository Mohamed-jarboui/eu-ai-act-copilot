.PHONY: install lint test ingest eval-retrieval serve examples calibrate eval-router traces

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

serve:
	uv run --env-file .env python -m copilot.api

examples:
	uv run --env-file .env python -m copilot.answer.examples

calibrate:
	uv run python -m copilot.answer.calibrate

eval-router:
	uv run --env-file .env python evals/run_router.py

traces:
	uv run --env-file .env python -m copilot.agent.traces
