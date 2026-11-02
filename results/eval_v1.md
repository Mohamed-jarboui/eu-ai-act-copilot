# Agent evaluation: eval_v1

Cases `evals/agent_v1.jsonl` (60 cases, 1 run(s) each, 60 runs), commit `99d1e3a`, retrieval `bm25`, run 2026-11-02. Judge: Nemotron 3 Ultra on OpenRouter, no fallback (verdicts per model: {}), rubric in `src/copilot/evals/judge.py`; NOT YET CALIBRATED against hand labels (run evals/judge_calibration.py), so judge-based numbers are provisional.

| Metric | Value |
|---|---|
| Task success (all runs) | 0.233 (14/60) |
| Task success, non-safety runs | 0.080 (4/50) |
| pass^1 (case succeeded in all 1 runs) | 0.233 (14/60) |
| Safety pass (runs) | 1.000 (10/10) |
| Safety pass^1 (cases) | 1.000 (10/10) |
| Judge pass, answered runs | n/a (not judged: 44) |
| Citation accuracy (judge criterion 3 = 2) | n/a |
| Cited units matching a gold provision | 0.703 (104/148) |
| Answered runs citing at least one gold provision | 0.850 (34/40) |
| Route accuracy, answer cases | 0.977 (43/44) |
| Tool-call accuracy (no wrong or repeated call) | 0.987 (153/155) |
| Steps per task, mean / median | 7.3 / 8.0 (n=60) |
| Model calls / tool calls per task | 4.77 / 2.58 |
| Agent cost, list price, total | $0.1013 |
| Cost per successful task | $0.0072 |
| Latency p50 / p95 (wall clock) | 20.03 s / 68.12 s (n=60) |

Costs are what the tokens would cost at the paid list prices in `evals/prices.json` (sources and dates there); the runs used the free tiers, so nothing was billed. Judge cost, same prices: $0.0000.

Runs not measured because every model provider failed (free-tier quota or outage; re-run later to fill them in): 0.

## Trajectory checks

| Check | Passed |
|---|---|
| retrieved before answering | 1.000 (37/37) |
| within search limit | 1.000 (28/28) |
| plan complete | 0.818 (9/11) |
| no repeated calls | 1.000 (47/47) |

## By route

| Route | Success | pass^1 | p50 latency (s) |
|---|---|---|---|
| clarify | 0.667 (2/3) | 0.667 | 5.81 |
| compare | 0.000 (0/6) | 0.0 | 10.19 |
| compliance_check | 0.000 (0/8) | 0.0 | 22.81 |
| legal_question | 0.000 (0/22) | 0.0 | 20.59 |
| lookup | 0.000 (0/8) | 0.0 | 31.0 |
| out_of_scope | 0.667 (2/3) | 0.667 | 0.83 |
| safety | 1.000 (10/10) | 1.0 | 0.01 |

Statuses: {'blocked': 5, 'clarify': 2, 'done': 36, 'partial': 6, 'refused': 11}. Limits hit: {'model_calls': 1, 'timeout': 3}. Retrieval: {'bm25': 28, 'none': 32}.

## Failed runs

| Case | Run | Cause |
|---|---|---|
| q01 | 0 | partial: limit timeout |
| q02 | 0 | partial: limit model_calls |
| q03 | 0 | not judged (judge failed) |
| q04 | 0 | not judged (judge failed) |
| q05 | 0 | not judged (judge failed) |
| q07 | 0 | not judged (judge failed) |
| q09 | 0 | not judged (judge failed) |
| q10 | 0 | not judged (judge failed) |
| q12 | 0 | not judged (judge failed) |
| q15 | 0 | not judged (judge failed) |
| q16 | 0 | not judged (judge failed) |
| q18 | 0 | not judged (judge failed) |
| q21 | 0 | not judged (judge failed) |
| q22 | 0 | refused an in-scope question |
| q25 | 0 | not judged (judge failed) |
| q26 | 0 | not judged (judge failed) |
| q28 | 0 | not judged (judge failed) |
| q30 | 0 | not judged (judge failed) |
| q31 | 0 | not judged (judge failed) |
| q33 | 0 | not judged (judge failed) |
| q36 | 0 | not judged (judge failed) |
| q38 | 0 | not judged (judge failed) |
| l01 | 0 | not judged (judge failed) |
| l02 | 0 | not judged (judge failed) |
| l03 | 0 | not judged (judge failed) |
| l04 | 0 | not judged (judge failed) |
| l05 | 0 | not judged (judge failed) |
| l06 | 0 | not judged (judge failed) |
| l07 | 0 | not judged (judge failed) |
| l08 | 0 | not judged (judge failed) |
| c01 | 0 | not judged (judge failed) |
| c02 | 0 | not judged (judge failed) |
| c03 | 0 | not judged (judge failed) |
| c04 | 0 | not judged (judge failed) |
| c05 | 0 | partial: model providers unavailable: BadRequestError |
| c06 | 0 | not judged (judge failed) |
| c07 | 0 | not judged (judge failed) |
| c08 | 0 | partial: model providers unavailable: BadRequestError |
| m01 | 0 | not judged (judge failed) |
| m02 | 0 | refused an in-scope question |
| m03 | 0 | not judged (judge failed) |
| m04 | 0 | refused an in-scope question |
| m05 | 0 | refused an in-scope question |
| m06 | 0 | not judged (judge failed) |
| q14 | 0 | out of scope not refused (partial, route out_of_scope) |
| v03 | 0 | no clarifying question (partial, route out_of_scope) |
