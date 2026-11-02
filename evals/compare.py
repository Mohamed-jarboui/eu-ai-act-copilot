"""Before/after table for docs/failures.md and the pull request.

    uv run python evals/compare.py results/eval_v1_before_run0.json results/eval_v1.json

Both reports must cover the same cases with the same number of runs (the "before" evaluation has
3 runs per case; its run 0 is scored on its own with
`run_agent.py --tag eval_v1_before_run0 --from-tag eval_v1_before --runs 1 --rescore`). Writes
results/eval_v1_compare.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from copilot.evals.metrics import summarize

OUT = Path("results/eval_v1_compare.md")
METRICS = [
    ("task_success", "Task success"),
    ("task_success_regular", "Task success, non-safety runs"),
    ("safety_pass", "Safety pass"),
    ("judge_pass", "Judge pass, answered runs"),
    ("citation_accuracy_judge", "Citation accuracy (judge)"),
    ("gold_cited", "Answered runs citing a gold provision"),
    ("route_accuracy", "Route accuracy"),
    ("tool_call_accuracy", "Tool-call accuracy"),
]
# The causes the three fixes target, recognised in each run's failure reason (docs/failures.md).
CAUSES = {
    "1. fallback tool call rejected (fixed)": lambda r: "BadRequestError" in (r["failure"] or ""),
    "2. search loop over the tool budget (fixed)": lambda r: (
        r["failure"] == "partial: limit tool_calls"
    ),
    "4. injected instruction obeyed (fixed)": lambda r: r["case_id"] == "s04" and not r["success"],
    "partial, other limits": lambda r: (
        (r["failure"] or "").startswith("partial: limit")
        and r["failure"] != "partial: limit tool_calls"
    ),
    "refused an in-scope question": lambda r: r["failure"] == "refused an in-scope question",
    "judge: failed a criterion": lambda r: (r["failure"] or "").startswith("judge:"),
    "not judged (judge unavailable)": lambda r: (r["failure"] or "").startswith("not judged"),
}


def cell(m: dict | None) -> str:
    if not m or m.get("value") is None:
        return "n/a"
    return f"{m['value']:.3f} ({m['hits']}/{m['n']})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before, after = (json.loads(p.read_text("utf-8")) for p in (args.before, args.after))
    # Only the cases measured on both sides (a run lost to the providers is not measured).
    common = {r["case_id"] for r in before["runs"]} & {r["case_id"] for r in after["runs"]}
    missing = sorted({r["case_id"] for r in before["runs"] + after["runs"]} - common)
    for report in (before, after):
        report["runs"] = [r for r in report["runs"] if r["case_id"] in common]
        report["summary"] = summarize(report["runs"], 1)
    b, a = before["summary"], after["summary"]
    lines = [
        "# Before / after the Stage 5 fixes",
        "",
        f"Same {a['cases']} cases, one run each"
        + (f" (left out, not measured on both sides: {', '.join(missing)})" if missing else "")
        + f". Before: `{args.before}` (run 0 of the 3-run evaluation; runs made with commit "
        f"{commits(before)}). After: `{args.after}` (runs made with commit {commits(after)}). "
        f"Same judge, same retrieval (`{after['retrieval']}`).",
        "",
        *(
            [
                f"> **Provisional:** {b['judge_missing']} answered runs before and "
                f"{a['judge_missing']} after have no verdict yet and count as failures, so task "
                "success, judge pass and citation accuracy are not comparable until both sides "
                "are judged. The other rows do not depend on the judge.",
                "",
            ]
            if b["judge_missing"] or a["judge_missing"]
            else []
        ),
        "| Metric | Before | After |",
        "|---|---|---|",
        *[f"| {label} | {cell(b.get(key))} | {cell(a.get(key))} |" for key, label in METRICS],
        f"| Steps per task, mean | {b['steps_per_task']['mean']} | {a['steps_per_task']['mean']} |",
        f"| Latency p50 / p95 (s) | {b['latency_s']['p50']} / {b['latency_s']['p95']} | "
        f"{a['latency_s']['p50']} / {a['latency_s']['p95']} |",
        f"| Cost per successful task (list price) | {_usd(b['cost_per_successful_task_usd'])} | "
        f"{_usd(a['cost_per_successful_task_usd'])} |",
        f"| Trajectory: plan complete | {cell(b['trajectory']['plan_complete'])} | "
        f"{cell(a['trajectory']['plan_complete'])} |",
        "",
        "## Failed runs by cause",
        "",
        "| Cause | Before | After |",
        "|---|---|---|",
        *[
            f"| {name} | {sum(map(test, before['runs']))} | {sum(map(test, after['runs']))} |"
            for name, test in CAUSES.items()
        ],  # fmt: skip
        "",
    ]
    OUT.write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))


def commits(report: dict) -> str:
    """The code the runs were made with (a rescored report carries the scoring commit)."""
    return ", ".join(f"`{c}`" for c in sorted({r.get("commit") or "?" for r in report["runs"]}))


def _usd(value) -> str:
    return "n/a" if value is None else f"${value:.4f}"


if __name__ == "__main__":
    main()
