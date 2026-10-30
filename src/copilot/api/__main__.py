"""`python -m copilot.api`: serve the API and the chat page on http://127.0.0.1:8000."""

from __future__ import annotations

import argparse

import uvicorn

from copilot.agent.factory import build_agent
from copilot.answer.factory import build_retriever, model_chains
from copilot.answer.impl import get_impl
from copilot.answer.pipeline import AnswerPipeline
from copilot.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    retriever = build_retriever()  # shared by /ask and /agent
    chains = model_chains()
    pipeline = AnswerPipeline(retriever, chains["generation"], chains["check"], get_impl())
    uvicorn.run(
        create_app(pipeline, build_agent(retriever=retriever)), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
