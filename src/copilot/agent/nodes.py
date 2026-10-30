"""The graph's nodes (docs/agent-design.md, section 1). Each takes (state, scope) and mutates
the scope; the wrapper in graph.py turns that into the state update and handles budgets."""

from __future__ import annotations

from types import SimpleNamespace

from copilot.agent import prompts
from copilot.agent.context import ALLOWED_TOOLS, LoopStop, Scope
from copilot.agent.guards import BLOCKED_MESSAGES, NOTICE, check_input, check_output, redact
from copilot.agent.router import parse_route, router_prompt, rule_route, settle
from copilot.agent.state import (
    LIMITS,
    MAX_CRITIC_ROUNDS,
    MAX_SEARCHES,
    MAX_STEP_TOOL_CALLS,
    MAX_TURNS,
    AgentState,
    SearchAttempt,
    SourceRef,
    StepResult,
    Turn,
)
from copilot.agent.tools import tool_catalog
from copilot.answer.evidence import Evidence
from copilot.answer.grounding import analyse, check_grounding
from copilot.answer.impl import get_impl
from copilot.answer.labels import label_for
from copilot.answer.prompt import NOT_COVERED, REFUSALS, SYSTEM, Source, build_prompt

IMPL = get_impl()  # the Stage 3 citation parser and refusal gate
MAX_ANSWER_SOURCES = 8
MAX_REPORT_SOURCES = 25
REPORT_SOURCE_CHARS = 1500


def to_source(s: SourceRef, limit: int | None = None) -> Source:
    return Source(
        s.cite_id, s.label, s.unit_id, s.regulation, s.title, s.text[:limit] if limit else s.text
    )


# --- guard and routing ---------------------------------------------------------------------


def input_guard(state: AgentState, scope: Scope) -> None:
    verdict = check_input(state.task.question)
    scope.task.sanitized_question = verdict.sanitized
    scope.task.injection_flagged = verdict.flagged
    scope.event(
        "input_guard",
        blocked=verdict.blocked,
        reason=verdict.reason,
        flagged=verdict.flagged,
        question=redact(state.task.question)[:300],
    )
    if verdict.blocked:
        scope.task.final_answer = BLOCKED_MESSAGES[state.task.language]
        scope.task.refusal_reason = verdict.reason
        scope.extra["status"] = "blocked"


def router(state: AgentState, scope: Scope) -> None:
    question = scope.task.sanitized_question
    decision = rule_route(question)
    if decision is None:
        turns = [f"{t.role}: {t.text[:200]}" for t in state.conversation.turns]
        text = scope.model("router", router_prompt(question, turns), max_output_tokens=400)
        decision = parse_route(text)
        if decision is None:  # unreadable: treat as a general legal question, low confidence
            decision = parse_route('{"route": "legal_question", "confidence": 0.5}')
    already_asked = (
        bool(state.conversation.turns) and state.conversation.turns[-1].kind == "clarify"
    )
    decision = settle(decision, question, already_asked)  # policy shared with evals/run_router.py
    clarify = decision.clarify
    scope.task.route = decision.route
    scope.task.route_confidence = decision.confidence
    scope.task.regulation_hint = decision.regulation
    scope.task.lookup_reference = decision.reference
    scope.task.clarifying_question = decision.clarifying_question if clarify else None
    scope.task.needs_clarification = clarify
    scope.extra["limits"] = LIMITS[decision.route]
    scope.event(
        "route",
        route=decision.route,
        confidence=decision.confidence,
        method=decision.method,
        regulation=decision.regulation,
        reference=label_for(decision.reference) if decision.reference else None,
        clarify=clarify,
    )


def clarify(state: AgentState, scope: Scope) -> None:
    scope.task.final_answer = state.task.clarifying_question or prompts.CLARIFY[state.task.language]
    scope.extra["status"] = "clarify"


def out_of_scope(state: AgentState, scope: Scope) -> None:
    scope.task.final_answer = prompts.OUT_OF_SCOPE[state.task.language]
    scope.task.refusal_reason = "out_of_scope"
    scope.extra["status"] = "refused"


# --- lookup ----------------------------------------------------------------------------------


def lookup(state: AgentState, scope: Scope) -> None:
    cid = state.task.lookup_reference
    regulation = cid.split(":")[0]
    label = label_for(cid)
    reference = label.split(" Art. ", 1)[1] if " Art. " in label else label.split(" ", 2)[2]
    result = scope.tool(
        "get_article",
        {"regulation": regulation, "reference": reference, "language": state.task.language},
    )
    scope.add_sources(result.sources)
    if not result.ok:
        scope.task.final_answer = REFUSALS[state.task.language]
        scope.task.refusal_reason = "not_found"
        scope.task.notes.append(result.error.message)
        scope.extra["status"] = "refused"


# --- agentic RAG loop ------------------------------------------------------------------------


def rewrite(state: AgentState, scope: Scope) -> None:
    previous = [q for s in state.task.searches for q in s.queries]
    reason = state.task.searches[-1].reason if state.task.searches else None
    prompt = prompts.rewrite_prompt(
        scope.task.sanitized_question,
        state.task.language,
        state.task.route or "legal_question",
        previous,
        reason,
    )
    data = prompts.parse_json(scope.model("rewrite", prompt, max_output_tokens=400)) or {}
    queries = []
    for item in data.get("queries", [])[:3]:
        if isinstance(item, dict) and isinstance(item.get("query"), str):
            reg = item.get("regulation") if item.get("regulation") in ("aiact", "gdpr") else "both"
            queries.append(f"{reg}|{item['query'].strip()}")
    if not queries:  # the model gave nothing usable: search with the question itself
        queries = [f"{state.task.regulation_hint or 'both'}|{scope.task.sanitized_question}"]
    scope.task.searches.append(SearchAttempt(queries=queries, chunk_ids=[]))
    scope.event("rewrite", queries=queries)


def search(state: AgentState, scope: Scope) -> None:
    attempt = scope.task.searches[-1]
    seen_before = {c for s in scope.task.searches[:-1] for c in s.chunk_ids}
    found: list[str] = []
    best = None
    for item in attempt.queries:
        regulation, query = item.split("|", 1)
        result = scope.tool(
            "search_regulation",
            {
                "query": query[:300],
                "regulation": regulation,
                "language": state.task.language,
                "k": 6,
            },
        )
        if not result.ok and result.error.error != "repeated_call":  # repeated: reuse it
            continue
        scope.add_sources(result.sources)
        found += [s.cite_id for s in result.sources]
        attempt.mode = result.output["mode"]
        evidence = result.output.get("evidence") or {}
        if best is None or (evidence.get("top_cosine") or -1) > (best.get("top_cosine") or -1):
            best = evidence
        scope.task.retrieval_notes = list(
            dict.fromkeys(scope.task.retrieval_notes + result.output["notes"])
        )
    attempt.chunk_ids = list(dict.fromkeys(found))
    scope.task.retrieval_mode = attempt.mode
    if state.task.route == "compare" and len(scope.task.searches) == 1:
        _compare_links(state, scope)
    if seen_before and set(attempt.chunk_ids) <= seen_before:
        attempt.reason = "no_progress"  # nothing new: the loop ends as if no searches were left
    scope.task.evidence = best


def _compare_links(state: AgentState, scope: Scope) -> None:
    """compare only: follow the explicit or curated AI Act -> GDPR link of the top AI Act hit."""
    top = next((s for s in scope.task.sources if s.regulation == "aiact"), None)
    if top is None or scope.used.tool_calls >= state.limits.max_tool_calls - 1:
        return
    article = top.cite_id.split(":")[2]
    links = scope.tool(
        "cross_reference",
        {"regulation": "aiact", "reference": article, "language": state.task.language},
    )
    other = (links.output or {}).get("other_regulation", []) if links.ok else []
    if other and scope.used.tool_calls < state.limits.max_tool_calls:
        reference = other[0]["label"].split(" Art. ", 1)[-1]
        fetched = scope.tool(
            "get_article",
            {"regulation": "gdpr", "reference": reference, "language": state.task.language},
        )
        if fetched.ok:
            scope.add_sources(fetched.sources)
            scope.task.searches[-1].chunk_ids += [s.cite_id for s in fetched.sources]


def grade(state: AgentState, scope: Scope) -> None:
    attempt = scope.task.searches[-1]
    by_id = {s.cite_id: s for s in scope.task.sources}
    candidates = [by_id[c] for c in attempt.chunk_ids if c in by_id][:12]
    evidence_data = state.task.evidence
    ev = (
        Evidence(**evidence_data)
        if isinstance(evidence_data, dict)
        else Evidence(
            mode=attempt.mode,
            n_sources=len(candidates),
            top_cosine=None,
            top_bm25=None,
            query_tokens=0,
            language=state.task.language,
        )
    )
    ev = Evidence(**{**ev.__dict__, "n_sources": len(candidates)})
    if IMPL.should_refuse(ev):
        scope.task.grade_decision = "not_covered"
        scope.event(
            "grade", decision="not_covered", reason="refusal gate", top_cosine=ev.top_cosine
        )
        return _not_covered(state, scope)
    searches_left = MAX_SEARCHES - len(scope.task.searches)
    if attempt.reason == "no_progress":
        searches_left = 0
    prompt = prompts.grade_prompt(scope.task.sanitized_question, candidates, searches_left)
    data = prompts.parse_json(scope.model("grade", prompt, max_output_tokens=400)) or {}
    relevant = [
        candidates[i - 1].cite_id
        for i in data.get("relevant", [])
        if isinstance(i, int) and 1 <= i <= len(candidates)
    ]
    decision = data.get("decision")
    if decision not in ("answer", "research", "not_covered"):
        decision = "answer" if candidates else "not_covered"
    if decision == "research" and searches_left <= 0:
        decision = "answer" if relevant else "not_covered"
    if decision == "answer" and not relevant:
        relevant = [c.cite_id for c in candidates[:MAX_ANSWER_SOURCES]]
    attempt.relevant_ids = relevant
    attempt.reason = data.get("reason") if decision == "research" else attempt.reason
    scope.task.grade_decision = decision
    scope.event(
        "grade",
        decision=decision,
        relevant=[label_for(c) for c in relevant],
        reason=data.get("reason"),
        searches_left=searches_left,
    )
    if decision == "not_covered":
        _not_covered(state, scope)


def _not_covered(state: AgentState, scope: Scope) -> None:
    scope.task.final_answer = REFUSALS[state.task.language]
    scope.task.refusal_reason = "not_covered"
    scope.extra["status"] = "refused"


def answer(state: AgentState, scope: Scope) -> None:
    task = scope.task
    by_id = {s.cite_id: s for s in task.sources}
    if task.searches:
        ids = list(dict.fromkeys(c for s in task.searches for c in s.relevant_ids))
        chosen = [by_id[c] for c in ids if c in by_id][:MAX_ANSWER_SOURCES]
    else:  # lookup: the fetched provision
        chosen = task.sources[:MAX_ANSWER_SOURCES]
    extra = {"lookup": prompts.LOOKUP_EXTRA, "compare": prompts.COMPARE_EXTRA}.get(task.route, "")
    prompt = build_prompt(task.sanitized_question, [to_source(s) for s in chosen], task.language)
    prompt += extra + prompts.feedback_extra(task.critic_feedback)
    text = scope.model("answer", prompt, system=SYSTEM, tier="large", max_output_tokens=3000)
    if NOT_COVERED in text:
        return _not_covered(state, scope)
    task.draft, task.draft_checked = text, False
    scope.event("draft", words=len(text.split()), sources=[s.label for s in chosen])


# --- plan and execute ------------------------------------------------------------------------


def plan(state: AgentState, scope: Scope) -> None:
    scope.task.plan = prompts.PLAN[state.task.language]
    scope.event("plan", steps=scope.task.plan)


def _react(state: AgentState, scope: Scope, step: str) -> StepResult:
    allowed = sorted(ALLOWED_TOOLS[f"step_{step}"])
    catalog = tool_catalog(allowed)
    scratch: list[str] = []
    collected: list[SourceRef] = []
    data: dict = {}
    calls, summary, status = 0, "", "done"
    try:
        for _ in range(MAX_STEP_TOOL_CALLS + 2):  # tool calls plus the final "done" reply
            prompt = prompts.react_prompt(
                step,
                state.task.language,
                scope.task.sanitized_question,
                prompts.step_context(scope.task.steps),
                catalog,
                scratch,
                MAX_STEP_TOOL_CALLS - calls,
            )
            action = prompts.parse_json(scope.model("react", prompt, max_output_tokens=800))
            if action is None:
                scratch.append("Your last reply was not valid JSON. Reply with ONLY JSON.")
                continue
            if "done" in action or calls >= MAX_STEP_TOOL_CALLS:
                summary = str(action.get("done") or summary or "")
                break
            name, args = action.get("tool"), action.get("args")
            if not isinstance(name, str) or not isinstance(args, dict):
                scratch.append('Reply with {"tool": "<name>", "args": {...}} or {"done": "..."}.')
                continue
            result = scope.tool(name, args)
            calls += 1
            collected += result.sources
            scope.add_sources(result.sources)
            if name == "classify_ai_risk" and result.ok:
                data = result.output
            scratch.append(
                f"Action: {name} {args}\nObservation:\n<<<\n{result.for_model()[:3000]}\n>>>"
            )
    except LoopStop:
        status = "partial"
        summary = summary or "Stopped: the same tool call was repeated."
    if not summary:
        status = "partial" if status == "done" else status
        summary = "No conclusion within the tool-call limit."
    return StepResult(
        step=step,
        status=status,
        summary=summary[:1200],
        sources=collected,
        assumptions=list(data.get("assumptions", [])),
        missing_facts=list(data.get("missing_facts", [])),
        data=data,
    )


def _step(step: str):
    def node(state: AgentState, scope: Scope) -> None:
        result = _react(state, scope, step)
        scope.task.steps.append(result)
        scope.event(
            "step",
            step=step,
            status=result.status,
            summary=result.summary[:300],
            sources=[s.label for s in result.sources][:10],
        )

    return node


step_classify = _step("classify")
step_obligations = _step("obligations")
step_gdpr_overlap = _step("gdpr_overlap")


def report(state: AgentState, scope: Scope) -> None:
    task = scope.task
    step_ids = list(dict.fromkeys(s.cite_id for st in task.steps for s in st.sources))
    by_id = {s.cite_id: s for s in task.sources}
    chosen = [by_id[c] for c in step_ids if c in by_id][:MAX_REPORT_SOURCES]
    if not chosen:  # nothing to cite: say so instead of writing an unsupported report
        task.final_answer = REFUSALS[task.language]
        task.refusal_reason = "insufficient_facts"
        task.notes.append("No step produced citable provisions.")
        scope.extra["status"] = "refused"
        return
    prompt = build_prompt(
        task.sanitized_question, [to_source(s, REPORT_SOURCE_CHARS) for s in chosen], task.language
    )
    prompt += prompts.report_extra(task.language, task.steps) + prompts.feedback_extra(
        task.critic_feedback
    )
    text = scope.model("report", prompt, system=SYSTEM, tier="large", max_output_tokens=3000)
    if NOT_COVERED in text:
        return _not_covered(state, scope)
    task.draft, task.draft_checked = text, False
    scope.event("draft", words=len(text.split()), sources=[s.label for s in chosen][:12])


# --- critic and output -----------------------------------------------------------------------


class _ScopeLLM:
    """Lets the Stage 3 check_grounding call a model through the scope (so it is budgeted)."""

    def __init__(self, scope: Scope) -> None:
        self.scope = scope

    def generate(self, prompt, system=None, tier="small", max_output_tokens=1000, **_):
        text = self.scope.model(
            "ground", prompt, system=system, tier=tier, max_output_tokens=max_output_tokens
        )
        return SimpleNamespace(text=text)


def _check_draft(state: AgentState, scope: Scope):
    """Run the Stage 3 citation validation and grounding check on the current draft."""
    task = scope.task
    sources = [to_source(s) for s in task.sources]
    sentences = analyse(task.draft or "", sources, IMPL.parse_citation)
    scope._check()  # the critic's reserve: raises BudgetExceeded if even that is gone
    return sentences, check_grounding(_ScopeLLM(scope), sentences, sources)


def ground(state: AgentState, scope: Scope) -> None:
    task = scope.task
    sentences, result = _check_draft(state, scope)
    invalid = any(s.invalid_citation for s in sentences)
    too_many = len(result.removed) * 2 > max(len(sentences), 1)
    scope.event(
        "ground",
        status=result.status,
        kept=len(result.kept),
        removed=len(result.removed),
        invalid_citation=invalid,
        round=task.critic_rounds,
    )
    if (too_many or invalid) and task.critic_rounds < MAX_CRITIC_ROUNDS:
        task.critic_rounds += 1
        task.critic_feedback = result.removed
        task.draft = None  # the edge after the critic sends the run back to answer or report
        return
    _accept(state, scope, result)


def _accept(state: AgentState, scope: Scope, result) -> None:
    task = scope.task
    task.draft_checked = True
    task.removed_sentences = result.removed
    task.grounding_check = result.status
    if not result.kept:
        task.final_answer = REFUSALS[task.language]
        task.refusal_reason = "ungrounded"
        task.grounded = False
        scope.extra["status"] = "refused"
        return
    task.grounded = result.status == "passed"
    by_id = {s.cite_id: s for s in task.sources}
    task.cited = [by_id[c] for c in result.verified_cites if c in by_id]
    task.final_answer = " ".join(result.kept)
    if task.route == "compliance_check":
        task.final_answer += _compliance_tail(task)


def _compliance_tail(task) -> str:
    """Assumptions, missing facts and the notice: built from step data, not generated."""
    h = prompts.HEADINGS[task.language]
    assumptions = list(dict.fromkeys(a for s in task.steps for a in s.assumptions))
    missing = list(dict.fromkeys(m for s in task.steps for m in s.missing_facts))
    skipped = [s.step for s in task.steps if s.status != "done"]
    parts = []
    if assumptions:
        parts.append(f"\n\n{h['assumptions']}:\n" + "\n".join(f"- {a}" for a in assumptions))
    if missing:
        parts.append(f"\n\n{h['missing']}:\n" + "\n".join(f"- {m}" for m in missing))
    if skipped:
        parts.append(f"\n\n{h['skipped']}: {', '.join(skipped)}")
    parts.append(f"\n\n{NOTICE[task.language]}")
    return "".join(parts)


def finalize_partial(state: AgentState, scope: Scope) -> None:
    """A limit was hit: return what is grounded, never an unchecked draft."""
    task = scope.task
    note = prompts.PARTIAL[task.language].format(limit=state.limit_hit or "model")
    if task.draft and not task.draft_checked:
        try:
            _, result = _check_draft(state, scope)
            if result.status == "unavailable":  # the critic could not run: nothing verified
                raise RuntimeError("critic unavailable")
            _accept(state, scope, result)
        except Exception:  # not even the reserve is left: withhold the draft
            task.final_answer = REFUSALS[task.language]
            task.refusal_reason = "unchecked_draft_withheld"
    elif not task.final_answer:
        if task.route == "compliance_check" and task.steps:
            done = [f"- {s.step}: {s.summary}" for s in task.steps if s.status == "done"]
            task.final_answer = (
                "\n".join(done) + _compliance_tail(task) if done else REFUSALS[task.language]
            )
        else:
            task.final_answer = REFUSALS[task.language]
            task.refusal_reason = task.refusal_reason or "limit"
    task.final_answer = f"{task.final_answer}\n\n{note}"
    scope.event("partial", limit=state.limit_hit)


def output(state: AgentState, scope: Scope) -> None:
    task = scope.task
    status = scope.extra.get("status") or state.status
    if status == "running":
        status = "done"
    problems = check_output(
        task.final_answer or "",
        task.route,
        len(task.cited),
        refused=status in ("refused", "clarify", "blocked") or not task.draft,
    )
    if problems:
        task.notes += problems
    kind = {"clarify": "clarify", "refused": "refusal", "blocked": "refusal"}.get(status, "answer")
    turns = state.conversation.turns + [
        Turn(role="user", text=task.question[:2000], kind="question"),
        Turn(role="assistant", text=(task.final_answer or "")[:4000], kind=kind),
    ]
    scope.extra["conversation"] = state.conversation.model_copy(
        update={"turns": turns[-MAX_TURNS:], "language": task.language}
    )
    scope.extra["status"] = status
    scope.event(
        "output",
        status=status,
        route=task.route,
        grounded=task.grounded,
        citations=[s.label for s in task.cited],
        limit=state.limit_hit,
        problems=problems,
    )
