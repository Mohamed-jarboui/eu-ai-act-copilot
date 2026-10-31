"""Judge calibration against the owner's manual labels.

    uv run python evals/judge_calibration.py --sheet       # write 30 answers to label (blind)
    uv run python evals/judge_calibration.py               # agreement once labels exist

1. `--sheet` picks 30 finished answers from an evaluation (one run per case, judged fails first
   so both verdicts are tested) and writes evals/judge_calibration_sheet.jsonl and a readable .md.
   The judge's verdicts are not in the sheet, so the labels are not influenced by them.
2. The owner copies the sheet to evals/judge_calibration.jsonl and fills "label" ("pass" or
   "fail", using the same rubric as the judge: src/copilot/evals/judge.py) and, optionally, the
   0-2 "scores" per criterion.
3. Without --sheet: agreement (share of equal labels), Cohen's kappa, the confusion matrix and,
   where scores were given, per-criterion agreement. Written to results/judge_agreement.{json,md}.
   The judge is trusted when agreement >= 0.8 and kappa >= 0.6 (our rule, stated in advance).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from copilot.evals.cases import load_cases
from copilot.evals.judge import CRITERIA
from copilot.evals.metrics import cohen_kappa

SHEET = Path("evals/judge_calibration_sheet.jsonl")
LABELS = Path("evals/judge_calibration.jsonl")
OUT_JSON, OUT_MD = Path("results/judge_agreement.json"), Path("results/judge_agreement.md")
SIZE, MIN_AGREEMENT, MIN_KAPPA = 30, 0.8, 0.6


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def verdicts(tag: str) -> dict[tuple[str, int], dict]:
    path = Path(f"results/runs/{tag}.judge.jsonl")
    rows = read_jsonl(path) if path.exists() else []
    return {(r["case_id"], r["run"]): r["judge"]["verdict"] for r in rows
            if r.get("judge") and r["judge"].get("verdict")}  # fmt: skip


def write_sheet(tag: str) -> None:
    cases = {c.id: c for c in load_cases()}
    judged = verdicts(tag)
    runs = {(r["case_id"], r["run"]): r for r in read_jsonl(Path(f"results/runs/{tag}.jsonl"))}
    # Every finished answer can be labelled, judged yet or not (missing verdicts are filled in
    # later); judged fails go first, up to half the sheet, so both verdicts are tested.
    finished = [k for k, r in runs.items() if r["final"] and r["final"]["status"] == "done"
                and cases[k[0]].expected == "answer" and not cases[k[0]].safety]  # fmt: skip
    rng = random.Random(5)  # fixed seed: the same evaluation gives the same sheet
    rng.shuffle(finished)
    fails = [k for k in finished if k in judged and not judged[k]["pass"]]
    picked, seen = [], set()
    for key in fails[: SIZE // 2] + [k for k in finished if k not in fails]:
        if key[0] not in seen and len(picked) < SIZE:  # one run per case
            picked.append(key)
            seen.add(key[0])
    rows = []
    for case_id, run in sorted(picked):
        case, final = cases[case_id], runs[(case_id, run)]["final"]
        rows.append({
            "case_id": case_id, "run": run, "lang": case.lang, "question": case.question,
            "gold": case.gold, "answer": final["answer"],
            "citations": [{"label": c["label"], "snippet": c["snippet"]}
                          for c in final["citations"]],
            "label": None, "scores": {name: None for name in CRITERIA}, "notes": "",
        })  # fmt: skip
    SHEET.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    md = ["# Judge calibration sheet", "",
          f"{len(rows)} answers from `results/runs/{tag}.jsonl`. Label each one pass or fail with "
          "the rubric in `src/copilot/evals/judge.py`; write the labels in "
          "`evals/judge_calibration.jsonl` (a copy of the .jsonl sheet).", ""]  # fmt: skip
    for r in rows:
        md += [f"## {r['case_id']} (run {r['run']}, {r['lang']})", "", f"**Q:** {r['question']}",
               "", f"Gold: {', '.join(r['gold']) or 'none'}", "",
               *[f"> {line}" if line else ">" for line in r["answer"].split("\n")], "",
               *[f"- {c['label']}: {c['snippet'][:200]}" for c in r["citations"]], ""]  # fmt: skip
    SHEET.with_suffix(".md").write_text("\n".join(md), "utf-8")
    print(f"wrote {len(rows)} answers to {SHEET} and {SHEET.with_suffix('.md')}")


def agreement(tag: str) -> None:
    if not LABELS.exists():
        raise SystemExit(f"{LABELS} not found: label the sheet first (see the module docstring)")
    judged = verdicts(tag)
    labels = [r for r in read_jsonl(LABELS) if r.get("label") in ("pass", "fail")]
    pairs = [(r, judged[(r["case_id"], r["run"])]) for r in labels
             if (r["case_id"], r["run"]) in judged]  # fmt: skip
    mine = [r["label"] for r, _ in pairs]
    judge = ["pass" if v["pass"] else "fail" for _, v in pairs]
    n = len(pairs)
    agree = round(sum(a == b for a, b in zip(mine, judge, strict=True)) / n, 3) if n else None
    kappa = cohen_kappa(mine, judge)
    confusion = Counter(f"owner {a} / judge {b}" for a, b in zip(mine, judge, strict=True))
    per_criterion = {}
    for name in CRITERIA:
        scored = [(r["scores"][name], v["scores"][name]) for r, v in pairs
                  if isinstance((r.get("scores") or {}).get(name), int)]  # fmt: skip
        if scored:
            per_criterion[name] = {
                "n": len(scored),
                "exact": round(sum(a == b for a, b in scored) / len(scored), 3),
                "within_1": round(sum(abs(a - b) <= 1 for a, b in scored) / len(scored), 3),
            }
    trusted = n > 0 and agree >= MIN_AGREEMENT and (kappa or 0) >= MIN_KAPPA
    labelled_by = sorted({r.get("labelled_by") or "the owner" for r, _ in pairs})
    report = {"tag": tag, "labelled_by": labelled_by, "n": n, "agreement": agree, "kappa": kappa,
              "trusted": trusted,
              "rule": f"agreement >= {MIN_AGREEMENT} and kappa >= {MIN_KAPPA}",
              "confusion": dict(confusion), "per_criterion": per_criterion}  # fmt: skip
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    lines = ["# Judge vs owner labels", "",
             f"{n} labelled answers from `{tag}`, labelled by: {'; '.join(labelled_by)}. "
             f"Agreement {agree}, Cohen's kappa {kappa}. "
             f"Trusted: **{trusted}** (rule: {report['rule']}).", "",
             "Confusion: " + ", ".join(f"{k}: {v}" for k, v in sorted(confusion.items())), "",
             *[f"- {k}: exact {v['exact']}, within 1 point {v['within_1']} (n={v['n']})"
               for k, v in per_criterion.items()], ""]  # fmt: skip
    OUT_MD.write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="eval_v1_before")
    parser.add_argument("--sheet", action="store_true")
    args = parser.parse_args()
    write_sheet(args.tag) if args.sheet else agreement(args.tag)


if __name__ == "__main__":
    main()
