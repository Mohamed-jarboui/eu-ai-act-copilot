"""The LangGraph graph (docs/agent-design.md, section 1) and the runner that streams it.

Every node runs inside a wrapper that checks the budget first and turns a limit or a provider
failure into status "partial", which every edge sends to finalize_partial. Threads persist in a
SQLite checkpointer, so a follow-up (for example the answer to a clarifying question) continues
the same conversation.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from copilot.agent import nodes
from copilot.agent.budget import limit_hit
from copilot.agent.context import BudgetExceeded, RunContext, Scope
from copilot.agent.state import LIMITS, STATE_TYPES, AgentState, BudgetUsed, TaskState
from copilot.answer.language import detect_language
from copilot.llm.base import ProviderError

CHECKPOINTS = Path("data/cache/agent.sqlite")
UNGUARDED = {"output", "finalize_partial"}  # must always be able to run
TERMINAL = {"done", "refused", "clarify", "blocked"}


def guarded(name: str, fn: Callable[[AgentState, Scope], None], ctx: RunContext):
    def run(state: AgentState) -> dict:
        if name not in UNGUARDED and state.status == "running":
            hit = limit_hit(state, name)
            if hit:
                scope = Scope(state, ctx, name)
                scope.event("limit", limit=hit)
                return scope.updates(count_step=False, status="partial", limit_hit=hit)
        scope = Scope(state, ctx, name)
        scope.event("node", name=name)
        try:
            fn(state, scope)
        except BudgetExceeded as exc:
            scope.event("limit", limit=exc.limit)
            return scope.updates(status="partial", limit_hit=exc.limit)
        except ProviderError as exc:  # every provider failed: a partial result, not a crash
            scope.task.notes.append(f"model providers unavailable: {type(exc).__name__}")
            scope.event("error", error=type(exc).__name__)
            return scope.updates(status="partial")
        return scope.updates()

    return run


def _after(next_node: str | Callable[[AgentState], str]):
    """Edge function: partial -> finalize, terminal status -> output, else the next node."""

    def edge(state: AgentState) -> str:
        if state.status == "partial":
            return "finalize_partial"
        if state.status in TERMINAL:
            return "output"
        return next_node(state) if callable(next_node) else next_node

    return edge


def _route(state: AgentState) -> str:
    if state.task.needs_clarification:
        return "clarify"
    return {
        "lookup": "lookup",
        "legal_question": "rewrite",
        "compare": "rewrite",
        "compliance_check": "plan",
        "out_of_scope": "out_of_scope",
    }[state.task.route]


def _after_grade(state: AgentState) -> str:
    return "rewrite" if state.task.grade_decision == "research" else "answer"


def _after_ground(state: AgentState) -> str:
    if state.task.final_answer is None and state.task.draft is None:  # regenerate
        return "report" if state.task.route == "compliance_check" else "answer"
    return "output"


EDGES: dict[str, Any] = {
    "input_guard": "router",
    "router": _route,
    "clarify": "output",
    "out_of_scope": "output",
    "lookup": "answer",
    "rewrite": "search",
    "search": "grade",
    "grade": _after_grade,
    "answer": "ground",
    "plan": "step_classify",
    "step_classify": "step_obligations",
    "step_obligations": "step_gdpr_overlap",
    "step_gdpr_overlap": "report",
    "report": "ground",
    "ground": _after_ground,
}
NODES: dict[str, Callable] = {
    name: getattr(nodes, name) for name in [*EDGES, "finalize_partial", "output"]
}


def build_graph(ctx: RunContext, checkpointer=None):
    graph = StateGraph(AgentState)
    for name, fn in NODES.items():
        graph.add_node(name, guarded(name, fn, ctx))
    graph.add_edge(START, "input_guard")
    targets = [*NODES, "output", "finalize_partial"]
    for name, nxt in EDGES.items():
        graph.add_conditional_edges(name, _after(nxt), {t: t for t in targets})
    graph.add_edge("finalize_partial", "output")
    graph.add_edge("output", END)
    return graph.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: Path = CHECKPOINTS) -> SqliteSaver:
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    allowed = [(t.__module__, t.__name__) for t in STATE_TYPES]
    serde = JsonPlusSerializer(allowed_msgpack_modules=allowed)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False), serde=serde)


class AgentAnswer(BaseModel):
    """What POST /agent returns in its final event."""

    run_id: str
    thread_id: str
    answer: str
    language: str
    route: str | None
    route_confidence: float | None
    status: str
    limit_hit: str | None
    refusal_reason: str | None
    grounded: bool | None
    grounding_check: str | None
    removed_sentences: list[str]
    citations: list[dict]
    plan: list[str]
    steps: list[dict]
    missing_facts: list[str]
    retrieval_mode: str | None
    retrieval_notes: list[str]
    notes: list[str]
    usage: dict


def to_answer(state: AgentState, thread_id: str) -> AgentAnswer:
    t, u = state.task, state.used
    return AgentAnswer(
        run_id=t.run_id,
        thread_id=thread_id,
        answer=t.final_answer or "",
        language=t.language,
        route=t.route,
        route_confidence=t.route_confidence,
        status=state.status,
        limit_hit=state.limit_hit,
        refusal_reason=t.refusal_reason,
        grounded=t.grounded,
        grounding_check=t.grounding_check,
        removed_sentences=t.removed_sentences,
        citations=[
            {
                "id": s.unit_id,
                "label": s.label,
                "regulation": s.regulation,
                "snippet": " ".join(s.text.split())[:300],
            }
            for s in t.cited
        ],
        plan=t.plan,
        steps=[{"step": s.step, "status": s.status, "summary": s.summary} for s in t.steps],
        missing_facts=list(dict.fromkeys(m for s in t.steps for m in s.missing_facts)),
        retrieval_mode=t.retrieval_mode,
        retrieval_notes=t.retrieval_notes,
        notes=t.notes,
        usage={
            "steps": u.steps,
            "model_calls": u.model_calls,
            "tool_calls": u.tool_calls,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "unreported_token_calls": u.unreported_token_calls,
            "elapsed_s": u.elapsed_s,
            "cost_usd": u.cost_usd,
            "models": u.models,
        },
    )


class AgentRunner:
    def __init__(self, ctx: RunContext, checkpointer=None) -> None:
        self.ctx = ctx
        self.app = build_graph(ctx, checkpointer)
        self._lock = threading.Lock()  # one run at a time: the checkpointer shares a connection

    def stream(
        self, question: str, language: str | None = None, thread_id: str | None = None
    ) -> Iterator[dict]:
        """Yield trace events as they happen, then {"type": "final", "data": AgentAnswer}."""
        thread_id = thread_id or uuid.uuid4().hex
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
        with self._lock:
            self.ctx.tool_cache.clear()
            prior = self.app.get_state(config).values if self.app.checkpointer else {}
            conversation = prior.get("conversation") if prior else None
            lang = language or detect_language(question)
            start = AgentState(
                task=TaskState(run_id=uuid.uuid4().hex, question=question, language=lang),
                used=BudgetUsed(started_at=time.time()),
                limits=LIMITS["legal_question"],
            )
            payload = {k: getattr(start, k) for k in AgentState.model_fields}
            if conversation is not None:
                payload["conversation"] = conversation
            last = None
            for mode, chunk in self.app.stream(payload, config, stream_mode=["custom", "values"]):
                if mode == "custom":
                    yield chunk
                else:
                    last = chunk
            final = AgentState.model_validate(last) if isinstance(last, dict) else last
            yield {"type": "final", "data": to_answer(final, thread_id)}

    def run(self, question: str, language: str | None = None, thread_id: str | None = None):
        """Run to completion: (AgentAnswer, trace events)."""
        events = list(self.stream(question, language, thread_id))
        return events[-1]["data"], events[:-1]
