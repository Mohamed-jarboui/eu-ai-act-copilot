"""Agent evaluation: run every case k times, judge the answers, check the traces, report.

Run with:
    uv run --env-file .env python evals/run_agent.py                         # 60 cases x 3 runs
    uv run --env-file .env python evals/run_agent.py --subset ci --runs 1 --tag eval_ci

Every run is appended to results/runs/<tag>.jsonl as soon as it ends (final answer, full trace,
tokens, wall time), and every verdict to results/runs/<tag>.judge.jsonl, so an interrupted run
(quota, network) resumes where it stopped. `--rescore` only recomputes the report from those files.

Retrieval is BM25 only by default: CI has no embedding cache, the free embedding quota is kept
for the daily document-embedding task, and French and German have no vectors yet anyway.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path

from copilot.agent.factory import build_agent
from copilot.answer.factory import build_retriever
from copilot.evals.cases import Case, answered, cites_gold, load_cases, safety_pass, task_success
from copilot.evals.judge import anchored, judge, judge_llm, safety_check, units_by_key
from copilot.evals.metrics import load_prices, model_cost, summarize
from copilot.evals.trajectory import check, tool_calls
from copilot.ingest.corpus import read_corpus
from copilot.retrieve.chunks import Corpus
from copilot.retrieve.vector import VectorIndex

RUNS_DIR, RESULTS = Path("results/runs"), Path("results")
STOP_AFTER_INFRA = 3  # consecutive runs lost to the providers: the daily quota is gone, stop


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def latest(rows: list[dict]) -> list[dict]:
    """The last record of each (case, run): a re-run replaces a run lost to the providers."""
    by_key = {(r["case_id"], r["run"]): r for r in rows}
    return list(by_key.values())


def infra_failure(run: dict) -> bool:
    """Lost to the model providers, not to the agent: every provider failed with a rate-limit,
    server or network error (FallbackLLM raises AllProvidersFailedError only for those; a bad
    request from our side is raised as itself and counts as an agent failure)."""
    final = run["final"]
    if final is None:
        return "AllProvidersFailedError" in (run["error"] or "")
    return final["status"] == "partial" and any(
        "AllProvidersFailedError" in note for note in final["notes"]
    )


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(units, retrieval: str):
    corpus = Corpus(units)
    retriever = build_retriever(corpus)
    if retrieval == "bm25":
        retriever.vectors = VectorIndex(corpus, {})
    return build_agent(Path("data/cache/eval_agent.sqlite"), retriever=retriever)


def run_cases(cases: list[Case], k: int, path: Path, units, retrieval: str) -> None:
    done = {(r["case_id"], r["run"]) for r in latest(read_jsonl(path)) if not infra_failure(r)}
    todo = [(c, i) for i in range(k) for c in cases if (c.id, i) not in done]
    if not todo:
        return
    agent = build(units, retrieval)
    lost = 0
    for n, (case, i) in enumerate(todo, start=1):
        start = time.perf_counter()
        error = None
        try:
            final, events = agent.run(case.question, None, thread_id=f"eval-{uuid.uuid4().hex}")
            final = final.model_dump()
        except Exception as exc:  # recorded as a failed run; the report counts it
            final, events = None, []
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        wall = round(time.perf_counter() - start, 2)
        at = datetime.now().isoformat(timespec="seconds")
        row = {"case_id": case.id, "run": i, "wall_s": wall, "final": final, "events": events,
               "error": error, "retrieval": retrieval, "commit": commit(), "at": at}  # fmt: skip
        append(path, row)
        status = final["status"] if final else "error"
        lost = lost + 1 if infra_failure(row) else 0
        print(f"[{n}/{len(todo)}] {case.id} run {i}: {status} route={final and final['route']} "
              f"{wall}s{' (providers unavailable)' if lost else ''}", flush=True)  # fmt: skip
        if lost >= STOP_AFTER_INFRA:
            print(f"stopped: {lost} runs in a row lost to the providers (quota?); rerun later "
                  "to resume", flush=True)  # fmt: skip
            return


def verdicts_complete(row: dict) -> bool:
    """A judged row counts only when every verdict it needed was given (a judge that failed on
    quota is asked again on the next run)."""
    return all(row.get(part) is None or row[part].get("verdict") is not None
               for part in ("judge", "safety"))  # fmt: skip


def judge_runs(
    cases: dict[str, Case], runs: list[dict], path: Path, units, limit: int | None = None
) -> None:
    done = {(r["case_id"], r["run"], r.get("at")) for r in read_jsonl(path) if verdicts_complete(r)}
    by_key = units_by_key(units)
    todo = []
    for r in runs:
        case, final = cases.get(r["case_id"]), r["final"]
        if case is None or final is None or (r["case_id"], r["run"], r["at"]) in done:
            continue
        # Only finished answers: a partial run fails task success whatever the judge says.
        finished = final["status"] == "done" and answered(final)
        needs_judge = case.expected == "answer" and not case.safety and finished
        needs_safety = bool(case.safety_rule) and finished
        if needs_judge or needs_safety:
            todo.append((case, r, needs_judge, needs_safety))
    # The hand-labelled answers first: the judge's agreement with them decides whether the other
    # verdicts can be trusted, so they are worth the scarce free-tier judge calls the most.
    labelled = Path("evals/judge_calibration.jsonl")
    first = {(x["case_id"], x["run"]) for x in read_jsonl(labelled)} if labelled.exists() else set()
    # then run 0 before runs 1 and 2: a one-run comparison needs only run 0
    todo.sort(key=lambda item: ((item[1]["case_id"], item[1]["run"]) not in first, item[1]["run"]))
    todo = todo[:limit] if limit is not None else todo
    failed = 0
    for n, (case, r, needs_judge, needs_safety) in enumerate(todo, start=1):
        if failed >= STOP_AFTER_INFRA:
            print(f"judge stopped: {failed} verdicts in a row failed (quota?); rerun later",
                  flush=True)  # fmt: skip
            return
        row = {"case_id": case.id, "run": r["run"], "at": r["at"], "judge": None, "safety": None}
        if needs_judge:
            row["judge"] = judge(judge_llm, case, r["final"], units, by_key)
        if needs_safety:
            row["safety"] = safety_check(judge_llm, case, r["final"])
        append(path, row)
        failed = 0 if verdicts_complete(row) else failed + 1
        v = (row["judge"] or {}).get("verdict") or {}
        print(f"[judge {n}/{len(todo)}] {case.id} run {r['run']}: pass={v.get('pass')} "
              f"{v.get('scores')}", flush=True)  # fmt: skip


def failure_reason(case: Case, row: dict) -> str | None:
    """One cause per failed run, for docs/failures.md."""
    if row["success"]:
        return None
    if row["status"] == "error":
        return "runner error"
    if case.safety:
        return f"safety: expected {case.expected}, got {row['status']}"
    if case.expected == "refuse":
        return f"out of scope not refused ({row['status']}, route {row['route']})"
    if case.expected == "clarify":
        return f"no clarifying question ({row['status']}, route {row['route']})"
    if row["status"] == "partial":
        if row["limit_hit"]:
            return f"partial: limit {row['limit_hit']}"
        return "partial: " + ("; ".join(row["notes"]) or "unchecked draft withheld")
    if row["status"] == "refused":
        return "refused an in-scope question"
    if row["status"] == "clarify":
        return "asked to clarify a clear question"
    if not row["route_ok"]:
        return f"wrong route ({row['route']})"
    if row["judge_pass"] is None:
        return "not judged (judge failed)"
    low = min(row["judge_scores"], key=lambda c: (row["judge_scores"][c], c != "correctness"))
    return f"judge: {low} scored {row['judge_scores'][low]}"


def score(cases: dict[str, Case], runs: list[dict], judged: list[dict], units) -> list[dict]:
    verdicts = {(j["case_id"], j["run"], j.get("at")): j for j in judged}
    prices = load_prices()
    all_by_id = {u.id: u for u in units if u.language == "en"}
    rows = []
    for r in runs:
        case = cases.get(r["case_id"])
        if case is None:
            continue
        final = r["final"] or {
            "status": "error", "route": None, "answer": "", "citations": [], "limit_hit": None,
            "retrieval_mode": None, "language": case.lang,
            "usage": {"steps": 0, "model_calls": 0, "cost_usd": None},
        }  # fmt: skip
        j = verdicts.get((case.id, r["run"], r["at"]), {})
        verdict = (j.get("judge") or {}).get("verdict")
        safety_verdict = (j.get("safety") or {}).get("verdict")
        safe = safety_pass(case, final, safety_verdict) if case.safety else None
        judge_ok = verdict["pass"] if verdict else None
        cited = [anchored(c["id"], all_by_id) for c in final["citations"]]
        n_tools, wrong = tool_calls(r["events"])
        usage = final["usage"]
        cost = usage.get("cost_usd")
        if cost is None and usage.get("model_calls", 0) == 0:
            cost = 0.0
        judge_cost = model_cost((j.get("judge") or {}).get("per_model", []) +
                                (j.get("safety") or {}).get("per_model", []), prices)  # fmt: skip
        row = {
            "case_id": case.id, "run": r["run"], "lang": case.lang, "safety": case.safety,
            "expected": case.expected, "route_expected": case.route, "gold": case.gold,
            "status": final["status"], "route": final["route"],
            "route_ok": final["route"] in case.accept_routes,
            "delivered": answered(final), "limit_hit": final.get("limit_hit"),
            "retrieval_mode": final.get("retrieval_mode"), "notes": final.get("notes", []),
            "cited": cited,
            "cited_match": [any(cites_gold(c, g) for g in case.gold) for c in cited],
            "judge_pass": judge_ok,
            "judge_scores": verdict["scores"] if verdict else None,
            "judge_quotes": (verdict["quotes_found"], verdict["quotes_total"]) if verdict else None,
            "safety_pass": safe,
            "trajectory": check(r["events"]),
            "tool_calls": n_tools, "wrong_tool_calls": wrong,
            "steps": usage.get("steps", 0), "model_calls": usage.get("model_calls", 0),
            "cost_usd": cost, "judge_cost_usd": judge_cost,
            "wall_s": r["wall_s"],
        }  # fmt: skip
        row["success"] = task_success(case, final, judge_ok, safe)
        row["failure"] = failure_reason(case, row)
        rows.append(row)
    return rows


def judge_models(judged: list[dict]) -> dict[str, int]:
    """Which model gave the verdicts (the last successful call of each judged run)."""
    counts: dict[str, int] = {}
    for j in judged:
        for part in ("judge", "safety"):
            models = (j.get(part) or {}).get("models") or []
            if models:
                counts[models[-1]] = counts.get(models[-1], 0) + 1
    return counts


def pct(m: dict) -> str:
    return "n/a" if m["value"] is None else f"{m['value']:.3f} ({m['hits']}/{m['n']})"


def markdown(tag: str, report: dict, rows: list[dict], k: int) -> str:
    s = report["summary"]
    t = s["trajectory"]
    lines = [
        f"# Agent evaluation: {tag}",
        "",
        f"Cases `{report['cases_file']}` ({s['cases']} cases, {k} run(s) each, {s['runs']} runs), "
        f"commit `{report['commit']}`, retrieval `{report['retrieval']}`, run {report['date']}. "
        "Judge: Nemotron 3 Ultra on OpenRouter, no fallback ("
        f"verdicts per model: {report['judge_models']}), rubric in "
        "`src/copilot/evals/judge.py`; " + report["judge_status"] + ".",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Task success (all runs) | {pct(s['task_success'])} |",
        f"| Task success, non-safety runs | {pct(s['task_success_regular'])} |",
        f"| pass^{k} (case succeeded in all {k} runs) | {pct(s[f'pass^{k}'])} |",
        f"| Safety pass (runs) | {pct(s['safety_pass'])} |",
        f"| Safety pass^{k} (cases) | {pct(s[f'safety_pass^{k}'])} |",
        f"| Judge pass, answered runs | {pct(s['judge_pass'])} "
        f"(not judged: {s['judge_missing']}) |",
        f"| Citation accuracy (judge criterion 3 = 2) | {pct(s['citation_accuracy_judge'])} |",
        f"| Cited units matching a gold provision | {pct(s['citation_precision_gold'])} |",
        f"| Answered runs citing at least one gold provision | {pct(s['gold_cited'])} |",
        f"| Route accuracy, answer cases | {pct(s['route_accuracy'])} |",
        f"| Tool-call accuracy (no wrong or repeated call) | {pct(s['tool_call_accuracy'])} |",
        f"| Steps per task, mean / median | {s['steps_per_task']['mean']} / "
        f"{s['steps_per_task']['median']} (n={s['steps_per_task']['n']}) |",
        f"| Model calls / tool calls per task | {s['model_calls_per_task']} / "
        f"{s['tool_calls_per_task']} |",
        f"| Agent cost, list price, total | {_usd(s['cost_usd_total'])} |",
        f"| Cost per successful task | {_usd(s['cost_per_successful_task_usd'])} |",
        f"| Latency p50 / p95 (wall clock) | {s['latency_s']['p50']} s / {s['latency_s']['p95']} s "
        f"(n={s['latency_s']['n']}) |",
        "",
        "Costs are what the tokens would cost at the paid list prices in `evals/prices.json` "
        "(sources and dates there); the runs used the free tiers, so nothing was billed. "
        f"Judge cost, same prices: {_usd(report['judge_cost_usd'])}.",
        "",
        f"Runs not measured because every model provider failed (free-tier quota or outage; "
        f"re-run later to fill them in): {len(report['not_measured'])}"
        + (f" ({', '.join(report['not_measured'])})" if report["not_measured"] else "")
        + ".",
        "",
        "## Trajectory checks",
        "",
        "| Check | Passed |",
        "|---|---|",
        *[f"| {name.replace('_', ' ')} | {pct(m)} |" for name, m in t.items()],
        "",
        "## By route",
        "",
        f"| Route | Success | pass^{k} | p50 latency (s) |",
        "|---|---|---|---|",
        *[
            f"| {name} | {pct(v['success'])} | {v['pass^k']} | {v['latency_p50_s']} |"
            for name, v in s["by_route"].items()
        ],  # fmt: skip
        "",
        f"Statuses: {s['statuses']}. Limits hit: {s['limits_hit'] or 'none'}. "
        f"Retrieval: {s['retrieval_modes']}.",
        "",
        "## Failed runs",
        "",
        "| Case | Run | Cause |",
        "|---|---|---|",
        *[f"| {r['case_id']} | {r['run']} | {r['failure']} |" for r in rows if r["failure"]],
        "",
    ]
    return "\n".join(lines)


def _usd(value) -> str:
    return "n/a (a model without a price)" if value is None else f"${value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["ci"], default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tag", default="eval_v1")
    parser.add_argument("--retrieval", choices=["bm25", "hybrid"], default="bm25")
    parser.add_argument("--only", nargs="*", help="case ids to run (default: all)")
    parser.add_argument("--rescore", action="store_true", help="no new runs or verdicts")
    parser.add_argument("--judge-only", action="store_true", help="no new runs, add verdicts")
    parser.add_argument("--judge-limit", type=int, help="at most this many runs judged now")
    parser.add_argument("--no-judge", action="store_true", help="runs only, no verdicts")
    parser.add_argument("--strict", action="store_true", help="exit 3 while runs are missing")
    parser.add_argument("--from-tag", help="score the run and verdict files of another tag")
    args = parser.parse_args()

    cases = load_cases(subset=args.subset)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    by_id = {c.id: c for c in cases}
    units = read_corpus()
    source = args.from_tag or args.tag
    runs_path = RUNS_DIR / f"{source}.jsonl"
    judge_path = RUNS_DIR / f"{source}.judge.jsonl"

    if not args.rescore and not args.judge_only:
        run_cases(cases, args.runs, runs_path, units, args.retrieval)
    recorded = [r for r in latest(read_jsonl(runs_path))
                if r["case_id"] in by_id and r["run"] < args.runs]  # fmt: skip
    lost = [r for r in recorded if infra_failure(r)]
    runs = [r for r in recorded if not infra_failure(r)]
    if not args.rescore and not args.no_judge:
        judge_runs(by_id, runs, judge_path, units, args.judge_limit)
    rows = score(by_id, runs, read_jsonl(judge_path), units)
    summary = summarize(rows, args.runs)
    agreement = RESULTS / "judge_agreement.json"
    judge_status = (
        "agreement with hand labels: see `results/judge_agreement.md` (and who labelled them)"
        if agreement.exists()
        else "NOT YET CALIBRATED against hand labels (run evals/judge_calibration.py), so "
        "judge-based numbers are provisional"
    )
    judge_costs = [r["judge_cost_usd"] for r in rows if r["judge_cost_usd"] is not None]
    report = {
        "tag": args.tag, "date": date.today().isoformat(), "commit": commit(),
        "cases_file": "evals/agent_v1.jsonl" + (" (ci subset)" if args.subset else ""),
        "retrieval": args.retrieval, "k": args.runs, "judge_status": judge_status,
        "judge_cost_usd": round(sum(judge_costs), 6),
        "judge_models": judge_models(read_jsonl(judge_path)),
        "not_measured": sorted(f"{r['case_id']}.{r['run']}" for r in lost),
        "summary": summary, "runs": rows,
    }  # fmt: skip
    out = RESULTS / f"{args.tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    md = markdown(args.tag, report, rows, args.runs)
    (RESULTS / f"{args.tag}.md").write_text(md, "utf-8")
    print(md)
    missing = len(cases) * args.runs - len(runs)
    print(f"runs missing (not yet run or lost to the providers): {missing}")
    if args.strict and missing:
        sys.exit(3)


if __name__ == "__main__":
    main()
