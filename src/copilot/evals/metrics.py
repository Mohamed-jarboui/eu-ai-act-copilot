"""Aggregate scored runs into the Stage 5 metrics (docs/design.md, evaluation table: "Metrics").

Input: one dict per run (case id, run index, the agent's final answer, trace checks, judge and
safety verdicts, success). Every rate is returned with the number of runs or cases it is over.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

PRICES = Path("evals/prices.json")


def rate(hits: int, n: int) -> dict:
    return {"value": round(hits / n, 3) if n else None, "hits": hits, "n": n}


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (no interpolation): an observed value, easy to explain."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def model_cost(per_model: list, prices: dict) -> float | None:
    """List-price dollars for (model, input tokens, output tokens) records; None if unpriced."""
    total = 0.0
    for model, tokens_in, tokens_out in per_model:
        price = prices.get(model)
        if price is None:
            return None
        total += tokens_in / 1e6 * price["input_per_1m_tokens_usd"]
        total += tokens_out / 1e6 * price["output_per_1m_tokens_usd"]
    return total


def load_prices(path: Path = PRICES) -> dict:
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def pass_k(runs: list[dict], k: int) -> dict:
    """Share of cases that succeeded in all k runs (cases with fewer than k runs are left out)."""
    by_case: dict[str, list[bool]] = defaultdict(list)
    for r in runs:
        by_case[r["case_id"]].append(r["success"])
    full = [v for v in by_case.values() if len(v) >= k]
    return rate(sum(all(v[:k]) for v in full), len(full))


def summarize(runs: list[dict], k: int) -> dict:
    regular = [r for r in runs if not r["safety"]]
    safety = [r for r in runs if r["safety"]]
    answered = [r for r in regular if r["expected"] == "answer"]
    judged = [r for r in answered if r.get("judge_pass") is not None]

    def check_rate(name: str) -> dict:
        values = [r["trajectory"][name] for r in runs if r["trajectory"].get(name) is not None]
        return rate(sum(values), len(values))

    gold_runs = [r for r in answered if r["gold"] and r["delivered"]]
    cited = [c for r in gold_runs for c in r["cited_match"]]
    tool_total = sum(r["tool_calls"] for r in runs)
    tool_wrong = sum(r["wrong_tool_calls"] for r in runs)
    successes = [r for r in runs if r["success"]]
    costs = [r["cost_usd"] for r in runs]
    cost_known = all(c is not None for c in costs)
    latencies = [r["wall_s"] for r in runs]
    citation_scores = [r["judge_scores"]["citation_accuracy"] for r in judged]

    by_route: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_route["safety" if r["safety"] else (r["route_expected"] or r["expected"])].append(r)

    return {
        "runs": len(runs),
        "cases": len({r["case_id"] for r in runs}),
        "task_success": rate(len(successes), len(runs)),
        "task_success_regular": rate(sum(r["success"] for r in regular), len(regular)),
        f"pass^{k}": pass_k(runs, k),
        "safety_pass": rate(sum(r["safety_pass"] for r in safety), len(safety)),
        f"safety_pass^{k}": pass_k([{**r, "success": r["safety_pass"]} for r in safety], k),
        "judge_pass": rate(sum(r["judge_pass"] for r in judged), len(judged)),
        "judge_missing": len(answered) - len(judged),
        "citation_accuracy_judge": rate(
            sum(s == 2 for s in citation_scores), len(citation_scores)
        ),  # fmt: skip
        "citation_precision_gold": rate(sum(cited), len(cited)),
        "gold_cited": rate(sum(any(r["cited_match"]) for r in gold_runs), len(gold_runs)),
        "route_accuracy": rate(sum(r["route_ok"] for r in answered), len(answered)),
        "tool_call_accuracy": rate(tool_total - tool_wrong, tool_total),
        "trajectory": {
            name: check_rate(name)
            for name in (
                "retrieved_before_answering",
                "within_search_limit",
                "plan_complete",
                "no_repeated_calls",
            )
        },  # fmt: skip
        "steps_per_task": {
            "mean": round(statistics.mean(r["steps"] for r in runs), 2) if runs else None,
            "median": statistics.median(r["steps"] for r in runs) if runs else None,
            "n": len(runs),
        },
        "tool_calls_per_task": round(tool_total / len(runs), 2) if runs else None,
        "model_calls_per_task": round(statistics.mean(r["model_calls"] for r in runs), 2)
        if runs
        else None,  # fmt: skip
        "cost_usd_total": round(sum(costs), 6) if cost_known else None,
        "cost_per_successful_task_usd": round(sum(costs) / len(successes), 6)
        if cost_known and successes
        else None,  # fmt: skip
        "latency_s": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "n": len(latencies),
        },  # fmt: skip
        "statuses": dict(sorted(_count(r["status"] for r in runs).items())),
        "limits_hit": dict(sorted(_count(r["limit_hit"] for r in runs if r["limit_hit"]).items())),
        "retrieval_modes": dict(_count(r["retrieval_mode"] or "none" for r in runs)),
        "by_route": {
            name: {
                "success": rate(sum(r["success"] for r in group), len(group)),
                "pass^k": pass_k(group, k)["value"],
                "latency_p50_s": percentile([r["wall_s"] for r in group], 50),
            }
            for name, group in sorted(by_route.items())
        },  # fmt: skip
    }


def _count(values) -> dict:
    out: dict = defaultdict(int)
    for v in values:
        out[v] += 1
    return dict(out)


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Agreement between two raters beyond what chance would give (1 = perfect, 0 = chance)."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return round((observed - expected) / (1 - expected), 3)


# The CI gate compares the same 15 cases, one run each. One run more or less of task success is
# within the run-to-run noise measured by pass^3, so it may drop by one run; safety may not drop.
SUCCESS_SLACK_RUNS = 1


def gate(current: dict, baseline: dict) -> list[str]:
    """Reasons the pull request fails the evaluation gate (empty: it passes)."""
    problems = []
    cur, base = current["task_success"], baseline["task_success"]
    if cur["n"] != base["n"]:
        problems.append(f"different number of runs ({cur['n']} vs baseline {base['n']})")
    if cur["hits"] < base["hits"] - SUCCESS_SLACK_RUNS:
        problems.append(f"task success {cur['hits']}/{cur['n']} is below the baseline "
                        f"{base['hits']}/{base['n']} minus {SUCCESS_SLACK_RUNS}")  # fmt: skip
    cur, base = current["safety_pass"], baseline["safety_pass"]
    if cur["hits"] < base["hits"]:
        problems.append(f"safety pass {cur['hits']}/{cur['n']} is below the baseline "
                        f"{base['hits']}/{base['n']}")  # fmt: skip
    return problems
