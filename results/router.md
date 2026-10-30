# Router evaluation

Labelled set `evals/routes_v0.jsonl` (labels drafted by the assistant, to be reviewed). Run 2026-10-30, clarification threshold 0.6.

- Accuracy: 0.905 on 74 labelled questions; 0.943 on the 70 not marked ambiguous.
- Routed by the explicit-reference rules, no model call: 8.
- Labelled questions that would get a clarifying question (confidence below the threshold): 0 of 74.
- Vague questions that got a clarifying question: 3 of 3.
- Median routing time: 1004 ms.

| Route | n | correct |
|---|---|---|
| compare | 6 | 6 |
| compliance_check | 8 | 8 |
| legal_question | 37 | 30 |
| lookup | 8 | 8 |
| out_of_scope | 15 | 15 |

Errors: legal_question -> compliance_check (7)
