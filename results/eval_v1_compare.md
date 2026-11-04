# Before / after the Stage 5 fixes

Same 60 cases, one run each. Before: `results\eval_v1_before_run0.json` (run 0 of the 3-run evaluation; runs made with commit `30701c8`). After: `results\eval_v1.json` (runs made with commit `5dc1661`, `7f178da`, `99d1e3a`). Same judge, same retrieval (`bm25`).

> **Provisional:** 0 finished answers before and 5 after are still waiting for a verdict and count as failures, so task success, judge pass and citation accuracy may still change. The other rows do not depend on the judge.

| Metric | Before | After |
|---|---|---|
| Task success | 0.567 (34/60) | 0.583 (35/60) |
| Task success, non-safety runs | 0.500 (25/50) | 0.500 (25/50) |
| Safety pass | 0.900 (9/10) | 1.000 (10/10) |
| Judge pass, answered runs | 0.613 (19/31) | 0.677 (21/31) |
| Citation accuracy (judge) | 0.774 (24/31) | 0.839 (26/31) |
| Answered runs citing a gold provision | 0.690 (29/42) | 0.850 (34/40) |
| Route accuracy | 0.977 (43/44) | 0.977 (43/44) |
| Tool-call accuracy | 0.976 (163/167) | 0.987 (153/155) |
| Steps per task, mean | 7.62 | 7.3 |
| Latency p50 / p95 (s) | 12.08 / 51.0 | 20.03 / 68.12 |
| Cost per successful task (list price) | $0.0030 | $0.0029 |
| Trajectory: plan complete | 0.727 (8/11) | 0.818 (9/11) |

## Failed runs by cause

| Cause | Before | After |
|---|---|---|
| 1. fallback tool call rejected (fixed) | 3 | 2 |
| 2. search loop over the tool budget (fixed) | 7 | 0 |
| 4. injected instruction obeyed (fixed) | 1 | 0 |
| partial, other limits | 1 | 2 |
| refused an in-scope question | 2 | 4 |
| judge: failed a criterion | 12 | 10 |
| not judged (judge unavailable) | 0 | 5 |
