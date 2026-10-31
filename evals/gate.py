"""CI gate: fail when task success or safety on the fast subset drops below the saved baseline.

    uv run python evals/gate.py results/eval_ci_pr.json            # compare (CI)
    uv run python evals/gate.py results/eval_ci.json --update      # save a new baseline

The baseline (results/eval_ci_baseline.json) is only ever written by hand with --update, after
a full evaluation, and is committed with the change that justified it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from copilot.evals.metrics import SUCCESS_SLACK_RUNS, gate

BASELINE = Path("results/eval_ci_baseline.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text("utf-8"))
    summary = report["summary"]
    if args.update:
        keep = {k: summary[k] for k in ("runs", "cases", "task_success", "safety_pass")}
        baseline = {"from": str(args.report), "commit": report["commit"], "date": report["date"],
                    "retrieval": report["retrieval"], **keep}  # fmt: skip
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", "utf-8")
        print(f"baseline saved: {json.dumps(keep)}")
        return 0
    baseline = json.loads(BASELINE.read_text("utf-8"))
    problems = gate(summary, baseline)
    for name in ("task_success", "safety_pass"):
        cur, base = summary[name], baseline[name]
        print(f"{name}: {cur['hits']}/{cur['n']} (baseline {base['hits']}/{base['n']})")
    print(f"allowed drop in task success: {SUCCESS_SLACK_RUNS} run; in safety: none")
    for p in problems:
        print(f"FAIL: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
