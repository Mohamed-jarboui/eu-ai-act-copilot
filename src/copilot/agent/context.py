"""What a node may do, in one place: call a model or a tool within the budget, and trace it.

A node gets a Scope: copies of the task, budget and trace that it mutates, then returns as its
state update. Every model call and tool call goes through the scope, which checks the budget
first (raising BudgetExceeded), enforces the per-node tool allow-list, detects repeated calls
and records trace events (also streamed live when the graph runs with stream_mode="custom").
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from copilot.agent.budget import call_key, limit_hit, record_model_calls, repeat_count
from copilot.agent.state import AgentState, LimitName, SourceRef, ToolCall, TraceEvent
from copilot.agent.tools import ToolError, ToolResult, Tools, run_tool

# Section 5 of the design: which tools each node may call.
ALLOWED_TOOLS: dict[str, set[str]] = {
    "lookup": {"get_article"},
    "search": {"search_regulation", "get_article", "get_definitions", "cross_reference"},
    "step_classify": {"classify_ai_risk", "get_article", "get_definitions"},
    "step_obligations": {"list_obligations", "get_article"},
    "step_gdpr_overlap": {"cross_reference", "search_regulation", "get_article"},
}


class BudgetExceeded(Exception):
    def __init__(self, limit: LimitName) -> None:
        super().__init__(limit)
        self.limit = limit


class LoopStop(Exception):
    """A third identical tool call: the current step stops."""


@dataclass
class Models:
    """One factory per purpose, each returning a fresh FallbackLLM-like object (usage list)."""

    factories: dict[str, Callable[[], Any]]

    def get(self, purpose: str):
        return self.factories[purpose]()


@dataclass
class RunContext:
    tools: Tools
    models: Models
    tool_cache: dict[str, ToolResult] = field(default_factory=dict)  # per run, by call key


class Scope:
    def __init__(self, state: AgentState, ctx: RunContext, node: str) -> None:
        self.state, self.ctx, self.node = state, ctx, node
        self.task = state.task.model_copy(deep=True)
        self.used = state.used.model_copy(deep=True)
        self.trace = list(state.trace)
        self.extra: dict[str, Any] = {}

    # --- tracing ----------------------------------------------------------------------------

    def event(self, kind: str, **data) -> None:
        event = TraceEvent(
            type=kind,
            node=self.node,
            at_ms=round((time.time() - self.used.started_at) * 1000, 1),
            data=data,
        )
        self.trace.append(event)
        try:  # live stream when running inside the graph with stream_mode="custom"
            from langgraph.config import get_stream_writer

            get_stream_writer()(event.model_dump())
        except Exception:
            pass

    # --- budget -----------------------------------------------------------------------------

    def _check(self) -> None:
        probe = self.state.model_copy(update={"used": self.used, "task": self.task})
        hit = limit_hit(probe, self.node)
        if hit:
            raise BudgetExceeded(hit)

    # --- model calls ------------------------------------------------------------------------

    def model(
        self,
        purpose: str,
        prompt: str,
        *,
        system: str | None = None,
        tier: str = "small",
        max_output_tokens: int = 1500,
    ) -> str:
        self._check()
        llm = self.ctx.models.get(purpose)
        start = time.perf_counter()
        try:
            text = llm.generate(
                prompt, system=system, tier=tier, max_output_tokens=max_output_tokens
            ).text
        finally:
            self.used = record_model_calls(self.used, llm, len(prompt) + len(system or ""), 0)
        self.event(
            "model_call",
            purpose=purpose,
            models=[f"{u.provider}:{u.model}" for u in llm.usage if u.error is None],
            failed=[u.error[:120] for u in llm.usage if u.error],  # attempts before the success
            ms=round((time.perf_counter() - start) * 1000),
        )
        return text

    # --- tool calls -------------------------------------------------------------------------

    def tool(self, name: str, args: dict) -> ToolResult:
        allowed = ALLOWED_TOOLS.get(self.node, set())
        if name not in allowed:
            result = ToolResult(
                error=ToolError(
                    error="not_allowed",
                    message=f"{name} is not available in this step; use one of: "
                    f"{', '.join(sorted(allowed)) or 'no tools'}.",
                )
            )
            self._record(name, args, result, 0.0)
            return result
        if (
            self.node == "step_gdpr_overlap"
            and name == "search_regulation"
            and args.get("regulation") != "gdpr"
        ):
            result = ToolResult(
                error=ToolError(
                    error="invalid_argument",
                    message='In this step search only the GDPR: set "regulation" to "gdpr".',
                )
            )
            self._record(name, args, result, 0.0)
            return result
        if name == "classify_ai_risk":  # classify what the user said, never a model's paraphrase
            own_words = self.user_description()
            if args.get("system_description") != own_words:
                self.event("guard", rule="classify the user's own description")
            args = {**args, "system_description": own_words}
        if name == "list_obligations" and (problem := self._obligations_problem(args)):
            result = ToolResult(error=problem)
            self._record(name, args, result, 0.0)
            return result
        key = call_key(name, args)
        seen = repeat_count(self.state.model_copy(update={"task": self.task}), key)
        if seen >= 2:
            self.event("loop", tool=name, args=args)
            raise LoopStop(name)
        if seen == 1 and key in self.ctx.tool_cache:
            earlier = self.ctx.tool_cache[key]
            result = ToolResult(
                output=earlier.output,
                error=ToolError(
                    error="repeated_call",
                    message="You already called this with the same arguments; the result is "
                    "above. Change the query or move on.",
                ),
                sources=earlier.sources,
            )
            self._record(name, args, result, 0.0, key=key)
            return result
        if self.used.tool_calls >= self.state.limits.max_tool_calls:
            raise BudgetExceeded("tool_calls")
        self._check()
        start = time.perf_counter()
        result = run_tool(self.ctx.tools, name, args)
        ms = (time.perf_counter() - start) * 1000
        if result.llm is not None:  # classify_ai_risk calls a model: charge it and trace it
            self.used = record_model_calls(self.used, result.llm, len(str(args)), 0)
            self.event(
                "model_call",
                purpose=f"tool:{name}",
                models=[f"{u.provider}:{u.model}" for u in result.llm.usage if u.error is None],
                failed=[u.error[:120] for u in result.llm.usage if u.error],
                ms=round(ms),
            )
        self.ctx.tool_cache[key] = result
        self._record(name, args, result, ms, key=key)
        return result

    def user_description(self) -> str:
        """The user's own words in this thread: earlier user turns, then the current question."""
        earlier = [t.text for t in self.state.conversation.turns if t.role == "user"]
        text = " ".join([*earlier[-2:], self.task.sanitized_question or self.task.question])
        return text[-3000:]

    def _obligations_problem(self, args: dict) -> ToolError | None:
        """Action guard: the category must come from classify_ai_risk, never from the model."""
        classified = next(
            (s.data for s in self.task.steps if s.step == "classify" and s.data.get("category")),
            None,
        )
        if classified is None:
            return ToolError(
                error="not_allowed",
                message="No risk classification is available, so no obligations can be listed. "
                "Finish this step and say the category could not be determined.",
            )
        allowed = {
            "high_risk": {"high_risk"},
            "limited": {"limited"},
        }.get(classified["category"], set())
        if classified.get("transparency_obligations"):
            allowed.add("limited")
        if classified.get("gpai"):
            allowed |= {"gpai", "gpai_systemic"}
        if args.get("category") not in allowed:
            options = ", ".join(sorted(allowed)) or "none: there are no obligations to list"
            return ToolError(
                error="invalid_argument",
                message=f"The classification says {classified['category']}; the category must "
                f"come from it. Allowed: {options}.",
            )
        return None

    def _record(self, name: str, args: dict, result: ToolResult, ms: float, key: str = "") -> None:
        self.used.tool_calls += 1
        self.task.tool_calls.append(
            ToolCall(
                node=self.node,
                name=name,
                args=args,
                key=key or call_key(name, args),
                ok=result.ok,
                error=result.error.error if result.error else None,
                latency_ms=round(ms, 1),
            )
        )
        self.event("tool_call", tool=name, args=args)
        summary = (
            {"error": result.error.error, "message": result.error.message}
            if result.error and result.error.error != "repeated_call"
            else {
                "ok": True,
                "summary": _summarize(name, result.output),
                "repeated": bool(result.error),
            }
        )
        self.event("tool_result", tool=name, ms=round(ms), **summary)

    def add_sources(self, sources: list[SourceRef]) -> None:
        known = {s.cite_id for s in self.task.sources}
        for s in sources:
            if s.cite_id not in known:
                self.task.sources.append(s)
                known.add(s.cite_id)

    # --- the state update -------------------------------------------------------------------

    def updates(self, count_step: bool = True, **extra) -> dict:
        """The node's state update. A step counts once the node has run, so checks made inside
        the node see the steps used before it."""
        if count_step:
            self.used.steps += 1
        self.used.elapsed_s = round(time.time() - self.used.started_at, 3)
        return {"task": self.task, "used": self.used, "trace": self.trace, **self.extra, **extra}


def _summarize(name: str, output: dict | None) -> str:
    if not output:
        return ""
    if name == "search_regulation":
        return ", ".join(r["label"] for r in output.get("results", [])[:6])
    if name in ("get_article", "cross_reference"):
        return output.get("label", "")
    if name == "get_definitions":
        return ", ".join(d["label"] for d in output.get("definitions", []))
    if name == "classify_ai_risk":
        return f"{output.get('category')} ({', '.join(output.get('basis', [])[:4])})"
    if name == "list_obligations":
        return f"{len(output.get('obligations', []))} provisions"
    return ""
