"""POST /agent: the server-sent event contract, with the agent running on fake models."""

import json

import pytest
from fastapi.testclient import TestClient

from agent_support import Script, build_runner, router_says
from answer_support import ANSWER, build_pipeline
from copilot.api.app import create_app


def events(text: str) -> list[tuple[str, dict]]:
    out = []
    for frame in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def client(corpus):
    pipeline, _, _ = build_pipeline(corpus, [ANSWER], ["[1, 1]"])
    return TestClient(create_app(pipeline, build_runner(Script())))


def test_agent_streams_trace_events_then_the_final_answer(client):
    response = client.post("/agent", json={"question": "Which AI practices are prohibited?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    stream = events(response.text)
    kinds = [kind for kind, _ in stream]
    assert kinds[-1] == "final" and set(kinds[:-1]) == {"trace"}
    trace_types = [data["type"] for kind, data in stream if kind == "trace"]
    assert "route" in trace_types and "tool_call" in trace_types and "tool_result" in trace_types
    final = stream[-1][1]
    assert {"answer", "route", "status", "grounded", "citations", "plan", "steps", "usage",
            "thread_id", "run_id", "limit_hit", "missing_facts"} <= set(final)  # fmt: skip
    assert final["route"] == "legal_question" and final["status"] == "done"


def test_the_routing_decision_is_streamed_with_its_confidence(client):
    stream = events(
        client.post("/agent", json={"question": "Which AI practices are prohibited?"}).text
    )
    route = next(d for k, d in stream if k == "trace" and d["type"] == "route")["data"]
    assert route["route"] == "legal_question" and route["confidence"] == 0.9


def test_a_thread_id_continues_the_conversation(corpus):
    pipeline, _, _ = build_pipeline(corpus, [ANSWER], ["[1, 1]"])
    script = Script(router=lambda p: router_says("legal_question", 0.3, question="Which one?"))
    client = TestClient(create_app(pipeline, build_runner(script)))
    first = events(client.post("/agent", json={"question": "Is it allowed?"}).text)[-1][1]
    assert first["status"] == "clarify"
    second = events(
        client.post(
            "/agent",
            json={"question": "Emotion recognition at work.", "thread_id": first["thread_id"]},
        ).text
    )[-1][1]
    assert second["thread_id"] == first["thread_id"] and second["status"] != "clarify"


@pytest.mark.parametrize(
    "body",
    [
        {"question": "hi"},
        {"question": "Which AI practices are prohibited?", "language": "es"},
        {"question": "Which AI practices are prohibited?", "thread_id": "../etc/passwd"},
    ],
)
def test_agent_validates_the_request(client, body):
    assert client.post("/agent", json=body).status_code == 422


def test_agent_not_configured_is_503(corpus):
    pipeline, _, _ = build_pipeline(corpus, [ANSWER], ["[1, 1]"])
    response = TestClient(create_app(pipeline)).post("/agent", json={"question": "What is X?"})
    assert response.status_code == 503


def test_health_says_whether_the_agent_is_available(client):
    assert client.get("/health").json()["agent"] is True


def test_the_page_has_the_reasoning_panel(client):
    page = client.get("/").text
    assert "Show reasoning steps" in page and 'fetch("/agent"' in page
