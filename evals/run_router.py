"""Router accuracy on the route-labelled questions (evals/routes_v0.jsonl), with the real model.

Run with:  uv run --env-file .env python evals/run_router.py

The labels were drafted by the assistant and are to be reviewed by the owner. Questions marked
"ambiguous" (they describe the asker's own situation and ask what applies) are reported with and
without; the 3 vague questions (route null, expect_clarify) measure clarification instead.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from copilot.agent.router import (
    CONFIDENCE_THRESHOLD,
    parse_route,
    router_prompt,
    rule_route,
    settle,
)
from copilot.answer.factory import model_chains

LABELS = Path("evals/routes_v0.jsonl")
OUT_JSON, OUT_MD = Path("results/router.json"), Path("results/router.md")


def main() -> None:
    rows = [json.loads(line) for line in LABELS.read_text("utf-8").splitlines() if line.strip()]
    small = model_chains()["small"]
    results = []
    for row in rows:
        start = time.perf_counter()
        decision = rule_route(row["question"])
        models: list[str] = []
        if decision is None:
            llm = small()
            text = llm.generate(router_prompt(row["question"], []), tier="small",
                                max_output_tokens=400).text  # fmt: skip
            models = [f"{u.provider}:{u.model}" for u in llm.usage if u.error is None]
            decision = parse_route(text)
            if decision:  # the same policy as the router node
                decision = settle(decision, row["question"], already_asked=False)
        results.append({
            **row,
            "predicted": decision.route if decision else None,
            "confidence": decision.confidence if decision else None,
            "method": decision.method if decision else "unparsed",
            "clarify": bool(decision) and decision.clarify,
            "ms": round((time.perf_counter() - start) * 1000),
            "models": models,
        })  # fmt: skip
        print(f"{row['id']:4} {row['route']!s:17} -> {results[-1]['predicted']!s:17} "
              f"{results[-1]['confidence']}", flush=True)  # fmt: skip

    labelled = [r for r in results if r["route"]]
    clear = [r for r in labelled if not r["ambiguous"]]
    vague = [r for r in results if r.get("expect_clarify")]

    def accuracy(group):
        return round(sum(r["predicted"] == r["route"] for r in group) / len(group), 3)

    per_route = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion = Counter()
    for r in labelled:
        per_route[r["route"]]["n"] += 1
        per_route[r["route"]]["correct"] += r["predicted"] == r["route"]
        if r["predicted"] != r["route"]:
            confusion[f"{r['route']} -> {r['predicted']}"] += 1
    report = {
        "date": date.today().isoformat(),
        "threshold": CONFIDENCE_THRESHOLD,
        "n_labelled": len(labelled),
        "accuracy": accuracy(labelled),
        "n_clear": len(clear),
        "accuracy_clear": accuracy(clear),
        "per_route": dict(per_route),
        "confusion": dict(confusion),
        "wrongly_clarified": sum(r["clarify"] for r in labelled),
        "vague_n": len(vague),
        "vague_clarified": sum(r["clarify"] for r in vague),
        "ms_p50": statistics.median(r["ms"] for r in results),
        "rules_routed": sum(r["method"] == "rules" for r in results),
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    lines = [
        "# Router evaluation",
        "",
        f"Labelled set `evals/routes_v0.jsonl` (labels drafted by the assistant, to be reviewed). "
        f"Run {report['date']}, clarification threshold {CONFIDENCE_THRESHOLD}.",
        "",
        f"- Accuracy: {report['accuracy']:.3f} on {report['n_labelled']} labelled questions; "
        f"{report['accuracy_clear']:.3f} on the {report['n_clear']} not marked ambiguous.",
        f"- Routed by the explicit-reference rules, no model call: {report['rules_routed']}.",
        f"- Labelled questions that would get a clarifying question (confidence below the "
        f"threshold): {report['wrongly_clarified']} of {report['n_labelled']}.",
        f"- Vague questions that got a clarifying question: {report['vague_clarified']} of "
        f"{report['vague_n']}.",
        f"- Median routing time: {report['ms_p50']} ms.",
        "",
        "| Route | n | correct |",
        "|---|---|---|",
        *[f"| {k} | {v['n']} | {v['correct']} |" for k, v in sorted(per_route.items())],
        "",
        "Errors: " + (", ".join(f"{k} ({v})" for k, v in confusion.most_common()) or "none"),
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
