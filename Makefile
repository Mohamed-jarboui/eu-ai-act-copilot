.PHONY: install lint test ingest eval-retrieval serve examples calibrate eval-router traces \n	eval-agent eval-ci judge-sheet judge-agreement

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

eval-agent:
	uv run --env-file .env python evals/run_agent.py

eval-ci:
	uv run --env-file .env python evals/run_agent.py --subset ci --runs 1 --tag eval_ci_pr
	uv run python evals/gate.py results/eval_ci_pr.json

judge-sheet:
	uv run python evals/judge_calibration.py --sheet

judge-agreement:
	uv run python evals/judge_calibration.py
