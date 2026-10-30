"""The router: the rule-based lookup check on the labelled set, model output parsing, and the
clarification policy. The model-based routing accuracy is measured live by evals/run_router.py."""

import json
from pathlib import Path

import pytest

from agent_support import Script, build_runner, router_says
from copilot.agent.router import CONFIDENCE_THRESHOLD, find_reference, parse_route, rule_route

LABELLED = [
    json.loads(line)
    for line in Path("evals/routes_v0.jsonl").read_text("utf-8").splitlines()
    if line.strip()
]


def test_labelled_set_has_every_route():
    routes = {r["route"] for r in LABELLED}
    assert routes >= {"lookup", "legal_question", "compliance_check", "compare", "out_of_scope"}
    assert len(LABELLED) == 77


@pytest.mark.parametrize(
    "row", [r for r in LABELLED if r["route"] == "lookup"], ids=lambda r: r["id"]
)
def test_every_labelled_lookup_is_caught_by_the_rules(row):
    decision = rule_route(row["question"])
    assert decision is not None and decision.route == "lookup" and decision.confidence == 1.0


@pytest.mark.parametrize(
    "row", [r for r in LABELLED if r["route"] != "lookup"], ids=lambda r: r["id"]
)
def test_the_rules_never_claim_a_non_lookup_question(row):
    assert rule_route(row["question"]) is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does Article 5 of the AI Act say?", ("aiact", "aiact:art:5")),
        ("Show me Article 22 GDPR.", ("gdpr", "gdpr:art:22")),
        ("Explain Article 6(3) of the AI Act.", ("aiact", "aiact:art:6:3")),
        ("Que dit l'article 6 du RGPD ?", ("gdpr", "gdpr:art:6")),
        ("Was steht in Artikel 17 Absatz 1 DSGVO?", ("gdpr", "gdpr:art:17:1")),
        ("Zeige mir Anhang IV der KI-Verordnung.", ("aiact", "aiact:anx:IV")),
        ("What is in Annex III, point 4 of the AI Act?", ("aiact", "aiact:anx:III:4")),
        ("Which AI practices are prohibited?", None),  # no named provision
    ],
)
def test_find_reference(question, expected):
    assert find_reference(question) == expected


def test_a_situation_mentioning_an_article_is_not_a_lookup():
    question = (
        "Under Article 6 of the AI Act, is our recruitment tool that ranks applicants for "
        "several clients across Europe considered high-risk, and what does that change for us?"
    )
    assert rule_route(question) is None


@pytest.mark.parametrize(
    ("text", "route", "confidence"),
    [
        (router_says("compare", 0.8), "compare", 0.8),
        ('Sure! {"route": "out_of_scope", "confidence": 1.4}', "out_of_scope", 1.0),  # clamped
        ('{"route": "legal_question", "confidence": "high"}', "legal_question", 0.0),
    ],
)
def test_parse_route(text, route, confidence):
    decision = parse_route(text)
    assert (decision.route, decision.confidence) == (route, confidence)


@pytest.mark.parametrize("text", ['{"route": "chit_chat"}', "no json", "{broken"])
def test_parse_route_rejects_unknown_output(text):
    assert parse_route(text) is None


# --- through the graph -------------------------------------------------------------------------


def test_low_confidence_asks_one_clarifying_question_then_routes():
    script = Script(router=lambda p: router_says("compliance_check", 0.3, question="Which system?"))
    runner = build_runner(script)
    first, _ = runner.run("What applies to us?", "en", thread_id="t-clarify")
    assert first.status == "clarify" and first.answer == "Which system?"
    assert CONFIDENCE_THRESHOLD > 0.3
    second, events = runner.run("A chatbot for our shop.", "en", thread_id="t-clarify")
    assert second.status != "clarify"  # at most one clarification per question
    assert any(e["type"] == "route" and not e["data"]["clarify"] for e in events)


def test_low_confidence_out_of_scope_asks_instead_of_refusing():
    """Vague questions come back as out_of_scope with low confidence: ask, do not refuse."""
    runner = build_runner(Script(router=lambda p: router_says("out_of_scope", 0.1)))
    answer, _ = runner.run("Is it allowed?", "en")
    assert answer.status == "clarify"


def test_confident_out_of_scope_is_refused():
    runner = build_runner(Script(router=lambda p: router_says("out_of_scope", 1.0)))
    answer, _ = runner.run("What is the VAT rate in France?", "en")
    assert answer.status == "refused" and answer.route == "out_of_scope"


def test_a_model_lookup_without_a_named_provision_becomes_a_legal_question():
    runner = build_runner(Script(router=lambda p: router_says("lookup", 0.95)))
    answer, _ = runner.run("Can I get a copy of all the personal data a company holds?", "en")
    assert answer.route == "legal_question" and answer.status == "done"


def test_unreadable_router_output_falls_back_to_a_legal_question():
    runner = build_runner(Script(router=lambda p: "I think this is about law"))
    answer, events = runner.run("Which AI practices are prohibited?", "en")
    route = next(e for e in events if e["type"] == "route")["data"]
    assert route["route"] == "legal_question" and route["confidence"] == 0.5
    assert route["clarify"] is True  # 0.5 is below the threshold
    assert answer.status == "clarify"


def test_lookup_needs_no_router_model_call():
    script = Script()
    answer, _ = build_runner(script).run("What does Article 22 GDPR say?", "en")
    assert answer.route == "lookup" and script.logs.get("router", []) == []
