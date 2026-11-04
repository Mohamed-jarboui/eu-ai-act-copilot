# Judge vs hand labels

30 labelled answers from `eval_v1_before`, labelled by: assistant (Claude), not the owner; 2026-10-31. Agreement 0.8, Cohen's kappa 0.587. Trusted: **False** (rule: agreement >= 0.8 and kappa >= 0.6).

Confusion: labels fail / judge fail: 8, labels pass / judge fail: 6, labels pass / judge pass: 16

- correctness: exact 0.733, within 1 point 0.933 (n=30)
- completeness: exact 0.8, within 1 point 1.0 (n=30)
- citation_accuracy: exact 0.767, within 1 point 0.967 (n=30)
- format: exact 0.9, within 1 point 0.967 (n=30)

Post-hoc, not the official result: under the rule 'pass when no criterion is 0 (post-hoc, to be confirmed on fresh labels)', agreement 0.9, kappa 0.754 (n=30).
