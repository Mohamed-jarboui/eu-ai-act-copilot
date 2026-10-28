# Design notes: agentic patterns applied to the EU AI Act Copilot

Source: "Agentic Design Patterns" guide (BASWE, 48 pages): 5 core patterns, 20 supporting patterns, design, evaluation, deployment, production checklist, interview questions.
Purpose: decide which patterns the copilot uses, and why, so every choice can be defended in an interview.

## 1. Patterns we use (and where)

| Pattern (guide section) | Where in the copilot | Why this one |
|---|---|---|
| **Workflow first** (1.4, 3.1) | Plain Q&A runs as a fixed chain: rewrite → retrieve → grade → answer → grounding check | Predictable, cheap, testable. Agent loops only where steps are unknown in advance |
| **Routing** (2.4) | First node classifies the request: *article lookup*, *legal question*, *compliance check*, *AI Act ↔ GDPR comparison*, *out of scope*. Low confidence → ask a clarifying question | Different request types need different paths and costs |
| **Agentic RAG** (3.5) | Retrieval is a controlled loop: rewrite query, hybrid search, grade relevance, re-retrieve if weak, answer with article citations | Fixes "confident wrong answer", the main RAG failure |
| **Reflection / critic** (2.3) | Grounding check: every claim must trace to a retrieved chunk; ungrounded claims are removed or the answer is regenerated (max 2 retries) | Deterministic-leaning critic with a retry cap, not open-ended self-critique |
| **Planning / plan-and-execute** (2.2, 3.2) | Compliance check ("we build a CV-screening tool, what applies?"): plan → classify risk category → collect obligations → check GDPR overlap → report | Structure is knowable up front, and the plan is shown to the user |
| **Tool use / ReAct** (2.1) | Inside each plan step, a short ReAct loop over tools | Steps depend on what the previous lookup found |
| **State machine / graph** (3.4) | Whole system as a LangGraph graph with typed state and checkpoints | Visible, testable, resumable control flow |
| **Guardrails** (3.4) | Input: prompt-injection and off-topic filter. Output: citation and grounding validator. Action: tool allow-list and argument validation | Required in production; cheap rules and validators first |
| **Termination & budgets** (3.4) | Max steps, token budget, timeout and cost ceiling per run; repeated-call detection; partial answer with a flag instead of a crash | Makes cost predictable and stops loops |
| **Context engineering** (4.1) | Lean tool outputs (article snippets, not whole chapters), static prompt prefix for caching, key constraints repeated at the end | Highest-leverage design skill per the guide |
| **Tool design** (4.2) | One job per tool, Pydantic schemas, informative errors, bounded output, all read-only | Most "reasoning" failures are tool failures |
| **MCP** (4.6) | Expose the copilot's tools as an MCP server, usable from Claude Desktop / IDEs | Interview topic; strong differentiator |

**Patterns we skip, on purpose:** multi-agent supervisor (one domain doesn't need it), tree search/LATS (too costly), debate, hierarchical agents, human-in-the-loop (all tools are read-only, so there are no irreversible actions). Being able to explain why they were *not* used is part of the story.

## 2. Tools (read-only)

`search_regulation(query, regulation, language, k)` · `get_article(regulation, number)` · `get_definitions(term)` · `classify_ai_risk(system_description)` → prohibited / high-risk (Annex III area) / limited / minimal, with the articles used · `list_obligations(role, risk_level)` · `cross_reference(article)` → related GDPR or AI Act articles

## 3. Evaluation (guide part 5)

| Level | Copilot implementation |
|---|---|
| Unit | Tool tests: right article for a number, correct schema errors, bounded output. Run on every commit |
| Retrieval | Hand-verified set mapping questions to correct articles: recall@k, MRR for BM25 vs vectors vs hybrid vs hybrid+rerank |
| Trajectory | Rules over traces: retrieved before answering, at most N searches, the compliance check visited every plan step |
| Outcome | Rubric judge (correctness, completeness, citations, format), calibrated against my own labels on 50+ cases |
| Safety | Injection attempts, out-of-scope and legal-advice-bait questions: pass = refuse or redirect correctly |
| Metrics | Task success with sample size, **pass^k** (k=3) for reliability, citation accuracy, tool-call accuracy, steps per task, cost per *successful* task, p50/p95 latency |
| CI gate | Eval subset in GitHub Actions; deploy blocked if success or safety drops |

Eval set: start with 40–60 cases across EN/FR/DE and all route types, versioned with the code, plus a held-out part. Every bug found later becomes a new case.

## 4. Production (guide parts 6–7)

Runs with a run ID, streaming progress, typed checkpoints in Postgres · Redis cache (embeddings, retrieval results, answers) · model tiering (small model for routing and grading, larger one for synthesis) · provider fallback (Gemini → Groq) with retries · OpenTelemetry tracing (Langfuse-compatible spans with tokens, cost, prompt version) · dashboard for success rate, p95 latency, cost per success · API keys, rate limits, audit log · secrets never in context, tool results treated as data · prompts, tools and model versioned together · kill switch via config flag.

**Lethal trifecta check:** the copilot reads only public law texts and has no external send capability, so the "exfiltration" leg is removed by design. This is worth stating in the README.

## 5. README must state (guide 7.5)

Pattern(s) used and why · architecture diagram · eval results with sample sizes · cost per run · one real failure and how it was fixed · what I would do differently.

## 6. Interview prep this project answers (guide 7.4)

Workflow vs agent · ReAct failure modes · planning vs ReAct · making reflection actually work · tool design · agent evaluation and pass^k · prompt injection and the trifecta · context engineering · cost control · MCP · "an agent you built and what broke".
