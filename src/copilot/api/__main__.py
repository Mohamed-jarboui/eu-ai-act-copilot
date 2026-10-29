"""`python -m copilot.api`: serve the API and the chat page on http://127.0.0.1:8000."""

from __future__ import annotations

import argparse

import uvicorn

from copilot.answer.factory import build_pipeline
from copilot.api.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(build_pipeline()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
