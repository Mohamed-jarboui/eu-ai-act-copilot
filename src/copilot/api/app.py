"""FastAPI app: POST /ask, POST /ask/stream, POST /agent (server-sent events), GET /health,
and the chat page."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from copilot.answer.models import Answer, AskRequest, Language
from copilot.answer.pipeline import AnswerPipeline
from copilot.llm.base import ProviderError

STATIC = Path(__file__).parent / "static"


def _sse(event: str, data) -> str:
    payload = data.model_dump_json() if hasattr(data, "model_dump_json") else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    language: Language | None = None
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Continue a conversation (for example after a clarifying question).",
    )


def create_app(pipeline: AnswerPipeline, agent=None) -> FastAPI:
    app = FastAPI(title="EU AI Act Copilot", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "retrieval_modes": pipeline.retriever.modes(),  # per language: hybrid or bm25
            "implementation": pipeline.impl.name,
            "agent": agent is not None,
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

    @app.post("/agent")
    def run_agent(request: AgentRequest) -> StreamingResponse:
        """Server-sent events: one `trace` event per step (routing decision, plan, tool calls and
        results, grading, critic, limits), then one `final` event with the answer. Only `final`
        carries the checked answer."""
        if agent is None:
            raise HTTPException(status_code=503, detail="the agent is not configured")

        def events() -> Iterator[str]:
            try:
                for event in agent.stream(request.question, request.language, request.thread_id):
                    if event.get("type") == "final":
                        yield _sse("final", event["data"])
                    else:
                        yield _sse("trace", event)
            except Exception as exc:  # report, do not drop the connection silently
                yield _sse("error", {"detail": f"agent failed: {type(exc).__name__}"})

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app
