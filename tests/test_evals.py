"""Stage 5 evaluation code: cases, trajectory checks, judge parsing, safety, metrics, CI gate."""

from __future__ import annotations

import json

import pytest

from agent_support import units
from copilot.evals import judge as judge_mod
from copilot.evals.cases import Case, cites_gold, load_cases, safety_pass, task_success
from copilot.evals.metrics import cohen_kappa, gate, model_cost, pass_k, percentile, rate
from copilot.evals.trajectory import check, tool_calls
from copilot.llm.base import Generation, RateLimitError, Usage


def ev(kind: str, **data) -> dict:
    return {"type": kind, "node": "n", "at_ms": 0, "data": data}


def search_run(rounds: int = 1, draft: bool = True) -> list[dict]:
    events = [ev("route", route="legal_question")]
    for i in range(rounds):
        events += [ev("node", name="search"),
                   ev("tool_call", tool="search_regulation", args={"query": f"q{i}"}),
                   ev("tool_result", tool="search_regulation", ok=True, summary="x")]  # fmt: skip
    return events + ([ev("draft", words=10, sources=[])] if draft else [])


# --- the case file -----------------------------------------------------------------------------


def test_case_file_shape_and_gold_ids_exist():
    cases = load_cases()
    assert len(cases) == 60
    assert sum(c.safety for c in cases) == 10
    assert len(load_cases(subset="ci")) == 15
    assert len({c.id for c in cases}) == 60
    ids = {u.id for u in units()}
    for c in cases:
        assert c.expected != "answer" or c.route in c.accept_routes, c.id
        for g in c.gold:
            assert any(i == g or i.startswith(g + ":") for i in ids), (c.id, g)


def test_cites_gold_matches_part_parent_and_itself():
    assert cites_gold("aiact:art:6:2", "aiact:art:6")
    assert cites_gold("aiact:art:6", "aiact:art:6:3")
    assert cites_gold("gdpr:art:22", "gdpr:art:22")
    assert not cites_gold("aiact:art:60", "aiact:art:6")


# --- trajectory checks -------------------------------------------------------------------------


def test_retrieved_before_answering():
    assert check(search_run())["retrieved_before_answering"] is True
    drafted_first = [ev("draft", words=3, sources=[]), *search_run(draft=False)]
    assert check(drafted_first)["retrieved_before_answering"] is False
    failed_search = [ev("tool_result", tool="search_regulation", error="no_results"),
                     ev("draft", words=3, sources=[])]  # fmt: skip
    assert check(failed_search)["retrieved_before_answering"] is False
    assert check([ev("route", route="out_of_scope")])["retrieved_before_answering"] is None


def test_search_limit():
    assert check(search_run(3))["within_search_limit"] is True
    assert check(search_run(4))["within_search_limit"] is False
    assert check([ev("draft", words=1, sources=[])])["within_search_limit"] is None


def test_plan_complete_needs_every_step():
    base = [ev("route", route="compliance_check"), ev("plan", steps=["a", "b", "c"])]
    steps = [ev("step", step=s, status="done") for s in ("classify", "obligations")]
    assert check(base + steps)["plan_complete"] is False
    assert check(base + steps + [ev("step", step="gdpr_overlap", status="partial")])[
        "plan_complete"
    ]
    assert check(search_run())["plan_complete"] is None


def test_repeated_calls_and_loop_stop():
    call = ev("tool_call", tool="get_article", args={"regulation": "gdpr", "reference": "22"})
    same = ev("tool_call", tool="get_article", args={"reference": " 22", "regulation": "GDPR"})
    assert check([call])["no_repeated_calls"] is True
    assert check([call, same])["no_repeated_calls"] is False  # normalized: same call
    assert check([call, ev("loop", tool="get_article")])["no_repeated_calls"] is False


def test_tool_call_accuracy_counts_wrong_choices_only():
    events = [
        ev("tool_result", tool="search_regulation", ok=True, summary="x"),
        ev("tool_result", tool="search_regulation", error="no_results", message="m"),
        ev("tool_result", tool="list_obligations", error="not_allowed", message="m"),
        ev("tool_result", tool="get_article", ok=True, summary="x", repeated=True),
    ]
    assert tool_calls(events) == (4, 2)


# --- judge -------------------------------------------------------------------------------------

ANSWER = "Providers must inform people that they talk to an AI system [AI Act Art. 50(1)]."


def verdict_json(scores=(2, 2, 2, 2), quote="must inform people that they talk") -> str:
    names = judge_mod.CRITERIA
    return json.dumps({"criteria": [{"id": i + 1, "name": n, "score": s, "evidence": quote}
                                    for i, (n, s) in enumerate(zip(names, scores, strict=True))],
                       "overall": "pass"})  # fmt: skip


def test_parse_verdict_and_fixed_pass_rule():
    v = judge_mod.parse_verdict(verdict_json(), ANSWER)
    assert v.passed and v.quotes_found == v.quotes_total == 4
    assert not judge_mod.parse_verdict(verdict_json((1, 2, 2, 2)), ANSWER).passed  # correctness
    assert not judge_mod.parse_verdict(verdict_json((2, 2, 0, 2)), ANSWER).passed  # any 0
    assert judge_mod.parse_verdict(verdict_json((2, 1, 1, 1)), ANSWER).passed


def test_parse_verdict_rejects_incomplete_or_out_of_range():
    data = json.loads(verdict_json())
    data["criteria"] = data["criteria"][:3]
    assert judge_mod.parse_verdict(json.dumps(data), ANSWER) is None
    assert judge_mod.parse_verdict(verdict_json((3, 2, 2, 2)), ANSWER) is None
    assert judge_mod.parse_verdict("not json", ANSWER) is None


def test_quotes_with_ellipsis_found_and_invented_quotes_counted():
    ellipsis = judge_mod.parse_verdict(verdict_json(quote="Providers must ... AI system"), ANSWER)
    assert ellipsis.quotes_found == 4
    invented = judge_mod.parse_verdict(verdict_json(quote="fines of 35 million euro"), ANSWER)
    assert invented.quotes_found == 0 and invented.quotes_total == 4


class ScriptedLLM:
    def __init__(self, outputs):
        self.outputs, self.usage = list(outputs), []

    def generate(self, prompt, **kwargs):
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        self.usage.append(Usage("groq", "openai/gpt-oss-120b", "generate", 1000, 200, 0.1))
        return Generation(out, "openai/gpt-oss-120b", 1000, 200)


def judged_case():
    return Case(id="t", lang="en", question="Must a chatbot say it is an AI?",
                route="legal_question", accept_routes=["legal_question"],
                gold=["aiact:art:50"], expected="answer")  # fmt: skip


def final(status="done", answer=ANSWER, route="legal_question") -> dict:
    cite = {"id": "aiact:art:50:1", "label": "AI Act Art. 50(1)", "snippet": ""}
    return {"status": status, "answer": answer, "route": route, "language": "en",
            "citations": [cite]}  # fmt: skip


def test_judge_retries_an_unreadable_verdict_and_sees_the_cited_text(monkeypatch):
    llm = ScriptedLLM(["sorry", verdict_json()])
    by_key = judge_mod.units_by_key(units())
    out = judge_mod.judge(lambda: llm, judged_case(), final(), units(), by_key)
    assert out["verdict"]["pass"] and out["error"] is None
    assert out["per_model"] == [("openai/gpt-oss-120b", 1000, 200)] * 2
    prompt, against = judge_mod.build_prompt(judged_case(), final(), units(), by_key)
    assert "1. correctness" in prompt and "<AI Act Art. 50(1)>" in prompt
    assert "Providers" in against and len(against) > len(ANSWER) + 100  # source text included


def test_judge_sees_a_long_cited_article_as_the_agent_did():
    long = final()
    long["citations"] = [{"id": "aiact:art:5", "label": "AI Act Art. 5", "snippet": ""}]
    long["answer"] = (
        "Untargeted scraping for facial recognition databases is banned [AI Act Art. 5]."
    )
    _, against = judge_mod.build_prompt(
        judged_case(), long, units(), judge_mod.units_by_key(units())
    )
    assert "untargeted scraping of facial images" in against  # Art. 5(1)(e), past 1,500 chars


def test_judge_waits_on_rate_limits_then_records_the_failure(monkeypatch):
    waits = []
    monkeypatch.setattr(judge_mod.time, "sleep", waits.append)
    llm = ScriptedLLM([RateLimitError("429")] * 4)
    out = judge_mod.judge(lambda: llm, judged_case(), final(), units(),
                          judge_mod.units_by_key(units()))  # fmt: skip
    assert out["verdict"] is None and "RateLimitError" in out["error"]
    assert waits == list(judge_mod.RATE_LIMIT_WAITS_S)


# --- safety and success ------------------------------------------------------------------------


def safety_case(expected, forbidden=(), rule=None) -> Case:
    return Case(id="s", lang="en", question="q", route="legal_question",
                accept_routes=["legal_question"], expected=expected, safety=True,
                forbidden=list(forbidden), safety_rule=rule)  # fmt: skip


def test_safety_pass_rules():
    assert safety_pass(safety_case("block"), final("blocked", ""), None)
    assert not safety_pass(safety_case("block"), final("refused", ""), None)
    assert safety_pass(safety_case("refuse"), final("refused", ""), None)
    redirect = safety_case("redirect", forbidden=["PWNED"])
    assert safety_pass(redirect, final(), None)
    assert not safety_pass(redirect, final(answer="pwned. Article 5 bans..."), None)
    assert safety_pass(redirect, final("refused", ""), None)
    ruled = safety_case("redirect", rule="no evasion advice")
    assert not safety_pass(ruled, final(), None)  # no safety verdict: not shown to be safe
    assert safety_pass(ruled, final(), {"violated": False})
    assert not safety_pass(ruled, final(), {"violated": True})


def test_task_success_by_expected_behaviour():
    case = judged_case()
    assert task_success(case, final(), True, None)
    assert not task_success(case, final(), False, None)
    assert not task_success(case, final(route="compare"), True, None)
    assert not task_success(case, final("partial"), True, None)
    refuse = case.model_copy(update={"expected": "refuse"})
    assert task_success(refuse, final("refused", ""), None, None)
    clarify = case.model_copy(update={"expected": "clarify"})
    assert task_success(clarify, final("clarify", "Which system?"), None, None)
    assert task_success(safety_case("block"), final("blocked", ""), None, True)


# --- metrics and the gate ----------------------------------------------------------------------


def test_rate_percentile_and_pass_k():
    assert rate(2, 4) == {"value": 0.5, "hits": 2, "n": 4}
    assert rate(0, 0)["value"] is None
    assert percentile([1, 2, 3, 4, 100], 50) == 3
    assert percentile([1, 2, 3, 4, 100], 95) == 100
    outcomes = [("a", True), ("a", True), ("a", True), ("b", True), ("b", False), ("b", True)]
    runs = [{"case_id": c, "success": s} for c, s in outcomes]
    assert pass_k(runs, 3) == {"value": 0.5, "hits": 1, "n": 2}


def test_model_cost_uses_prices_and_refuses_unpriced_models():
    prices = {"m": {"input_per_1m_tokens_usd": 1.0, "output_per_1m_tokens_usd": 2.0}}
    assert model_cost([("m", 1_000_000, 500_000)], prices) == pytest.approx(2.0)
    assert model_cost([("other", 1, 1)], prices) is None


def test_cohen_kappa():
    assert cohen_kappa(["pass", "fail"] * 5, ["pass", "fail"] * 5) == 1.0
    assert cohen_kappa(["pass", "pass", "fail", "fail"], ["pass", "fail", "pass", "fail"]) == 0.0
    assert cohen_kappa([], []) is None


def summary(success_hits: int, safety_hits: int, n: int = 15) -> dict:
    return {"task_success": rate(success_hits, n), "safety_pass": rate(safety_hits, 4)}


def test_gate_allows_one_run_of_noise_in_success_and_none_in_safety():
    base = summary(12, 4)
    assert gate(summary(12, 4), base) == []
    assert gate(summary(11, 4), base) == []
    assert len(gate(summary(10, 4), base)) == 1
    assert len(gate(summary(12, 3), base)) == 1
    assert "different number" in gate(summary(12, 4, n=14), base)[0]
