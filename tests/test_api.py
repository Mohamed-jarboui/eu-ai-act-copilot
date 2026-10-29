"""The HTTP contract of the API, with the pipeline running on fake models."""

import json
import re

import pytest
from fastapi.testclient import TestClient

from answer_support import ANSWER, HIGH_RISK_Q, build_pipeline
from copilot.api.app import create_app
from copilot.llm.base import AllProvidersFailedError


@pytest.fixture
def client(corpus):
    pipeline, _, _ = build_pipeline(corpus, [ANSWER], ["[1, 1]"])
    return TestClient(create_app(pipeline))


def parse_sse(text: str) -> list[tuple[str, object]]:
    events = []
    for frame in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["retrieval_modes"] == {"en": "bm25", "fr": "bm25", "de": "bm25"}
    assert body["implementation"] == "reference"


def test_ask_contract(client):
    response = client.post("/ask", json={"question": HIGH_RISK_Q})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "answer", "language", "refused", "refusal_reason", "citations", "grounded",
        "grounding_check", "removed_sentences", "retrieval_mode", "retrieval_notes", "usage",
    }  # fmt: skip
    assert body["language"] == "en" and body["refused"] is False and body["grounded"] is True
    assert set(body["citations"][0]) == {"id", "label", "regulation", "snippet"}
    assert set(body["usage"]) == {
        "input_tokens", "output_tokens", "model_calls", "models", "retrieval_ms", "generation_ms",
        "grounding_ms", "total_ms",
    }  # fmt: skip


def test_ask_validates_the_request(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422  # too short
    assert client.post("/ask", json={"question": HIGH_RISK_Q, "language": "es"}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_maps_provider_failure_to_503(corpus):
    pipeline, _, _ = build_pipeline(corpus, [AllProvidersFailedError("all down")])
    response = TestClient(create_app(pipeline)).post("/ask", json={"question": HIGH_RISK_Q})
    assert response.status_code == 503 and "unavailable" in response.json()["detail"]


def test_ask_stream_sends_sources_deltas_then_the_verified_answer(client):
    response = client.post("/ask/stream", json={"question": HIGH_RISK_Q})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)

    kinds = [kind for kind, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "final" and "delta" in kinds
    final = events[-1][1]
    assert final["answer"] == ANSWER and final["grounded"] is True
    assert set(final) >= {"answer", "citations", "grounded", "usage"}  # same shape as /ask


def test_ask_stream_reports_provider_failure_as_an_event(corpus):
    pipeline, _, _ = build_pipeline(corpus, [AllProvidersFailedError("all down")])
    response = TestClient(create_app(pipeline)).post("/ask/stream", json={"question": HIGH_RISK_Q})
    events = parse_sse(response.text)
    assert events[-1][0] == "error" and "unavailable" in events[-1][1]["detail"]


def test_the_chat_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "EU AI Act Copilot" in response.text
    script = response.text.split("<script>")[1]
    assert not re.search(r"\.innerHTML|insertAdjacentHTML|document\.write", script)  # never HTML
