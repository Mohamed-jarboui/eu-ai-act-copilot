# Before / after the Stage 5 fixes

Same 60 cases, one run each. Before: `results\eval_v1_before_run0.json` (run 0 of the 3-run evaluation; runs made with commit `30701c8`). After: `results\eval_v1.json` (runs made with commit `5dc1661`, `7f178da`, `99d1e3a`). Same judge, same retrieval (`bm25`).

> **Provisional:** 28 answered runs before and 44 after have no verdict yet and count as failures, so task success, judge pass and citation accuracy are not comparable until both sides are judged. The other rows do not depend on the judge.

| Metric | Before | After |
|---|---|---|
| Task success | 0.400 (24/60) | 0.233 (14/60) |
| Task success, non-safety runs | 0.300 (15/50) | 0.080 (4/50) |
| Safety pass | 0.900 (9/10) | 1.000 (10/10) |
| Judge pass, answered runs | 0.562 (9/16) | n/a |
| Citation accuracy (judge) | 0.750 (12/16) | n/a |
| Answered runs citing a gold provision | 0.690 (29/42) | 0.850 (34/40) |
| Route accuracy | 0.977 (43/44) | 0.977 (43/44) |
| Tool-call accuracy | 0.976 (163/167) | 0.987 (153/155) |
| Steps per task, mean | 7.62 | 7.3 |
| Latency p50 / p95 (s) | 12.08 / 51.0 | 20.03 / 68.12 |
| Cost per successful task (list price) | $0.0043 | $0.0072 |
| Trajectory: plan complete | 0.727 (8/11) | 0.818 (9/11) |

## Failed runs by cause

| Cause | Before | After |
|---|---|---|
| 1. fallback tool call rejected (fixed) | 3 | 2 |
| 2. search loop over the tool budget (fixed) | 7 | 0 |
| 4. injected instruction obeyed (fixed) | 1 | 0 |
| partial, other limits | 1 | 2 |
| refused an in-scope question | 2 | 4 |
| judge: failed a criterion | 7 | 0 |
| not judged (judge unavailable) | 15 | 36 |
