"""FastAPI app: POST /ask, POST /ask/stream (server-sent events), GET /health, and the chat page."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from copilot.answer.models import Answer, AskRequest
from copilot.answer.pipeline import AnswerPipeline
from copilot.llm.base import ProviderError

STATIC = Path(__file__).parent / "static"


def _sse(event: str, data) -> str:
    payload = data.model_dump_json() if hasattr(data, "model_dump_json") else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def create_app(pipeline: AnswerPipeline) -> FastAPI:
    app = FastAPI(title="EU AI Act Copilot", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "retrieval_modes": pipeline.retriever.modes(),  # per language: hybrid or bm25
            "implementation": pipeline.impl.name,
        }

    @app.post("/ask", response_model=Answer)
    def ask(request: AskRequest) -> Answer:
        try:
            return pipeline.ask(request.question, request.language)
        except ProviderError as exc:
            raise HTTPException(
                status_code=503, detail=f"model providers unavailable: {exc}"
            ) from exc

    @app.post("/ask/stream")
    def ask_stream(request: AskRequest) -> StreamingResponse:
        """Server-sent events: `sources`, then `delta` (draft text) events, then one `final`.

        The draft is NOT verified: only the `final` event carries the checked answer, which may
        drop sentences the grounding check could not support. Clients should replace the draft.
        """

        def events() -> Iterator[str]:
            try:
                for kind, payload in pipeline.events(
                    request.question, request.language, stream=True
                ):
                    yield _sse(kind, payload)
            except ProviderError as exc:
                yield _sse("error", {"detail": f"model providers unavailable: {exc}"})

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app
