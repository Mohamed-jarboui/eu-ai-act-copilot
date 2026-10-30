"""Typed agent state (docs/agent-design.md, section 2).

Three parts with different lifetimes: the conversation (persisted per thread by the SQLite
checkpointer), the task (one run, replaced at the start of the next run) and the budget (one run).
Lists are replaced, never appended through a reducer: a reducer would carry a run's trace into the
next run of the same thread.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

Route = Literal["lookup", "legal_question", "compliance_check", "compare", "out_of_scope"]
Status = Literal["running", "done", "clarify", "refused", "partial", "blocked"]
LimitName = Literal[
    "steps", "model_calls", "tool_calls", "tokens", "timeout", "cost", "repeated_call"
]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    text: str  # assistant turns: the final answer only, no traces
    kind: Literal["question", "answer", "clarify", "refusal"] = "answer"


class ConversationState(BaseModel):
    turns: list[Turn] = Field(default_factory=list)  # capped to the last MAX_TURNS
    language: str | None = None


class SourceRef(BaseModel):
    """Anything a sentence may cite."""

    cite_id: str  # "aiact:art:6:2"
    label: str  # "AI Act Art. 6(2)"
    unit_id: str
    regulation: str
    title: str | None = None
    text: str  # bounded (6000 chars, as in Stage 3)


class SearchAttempt(BaseModel):
    queries: list[str]
    chunk_ids: list[str]
    relevant_ids: list[str] = Field(default_factory=list)
    reason: str | None = None  # the grader's reason when it asked for a re-search
    mode: str = "bm25"


class StepResult(BaseModel):
    step: Literal["classify", "obligations", "gdpr_overlap"]
    status: Literal["done", "partial", "skipped", "failed"]
    summary: str
    sources: list[SourceRef] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)  # e.g. the risk classification


class ToolCall(BaseModel):
    node: str
    name: str
    args: dict[str, Any]
    key: str  # name + normalized arguments, for repeated-call detection
    ok: bool
    error: str | None = None
    latency_ms: float = 0.0


class TaskState(BaseModel):
    run_id: str = ""
    question: str = ""
    language: str = "en"
    sanitized_question: str = ""  # suspicious spans removed (input guard)
    injection_flagged: bool = False
    route: Route | None = None
    route_confidence: float | None = None
    regulation_hint: Literal["aiact", "gdpr", "both"] | None = None
    clarifying_question: str | None = None
    needs_clarification: bool = False
    lookup_reference: str | None = None  # "aiact:art:6:2" when the question names a provision
    searches: list[SearchAttempt] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)  # all the answer may cite
    retrieval_mode: str | None = None
    retrieval_notes: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] | None = None  # best retrieval signals of the last search
    grade_decision: Literal["answer", "research", "not_covered"] | None = None
    plan: list[str] = Field(default_factory=list)
    steps: list[StepResult] = Field(default_factory=list)
    draft: str | None = None
    draft_checked: bool = False
    critic_rounds: int = 0
    critic_feedback: list[str] = Field(default_factory=list)
    removed_sentences: list[str] = Field(default_factory=list)
    grounded: bool | None = None
    grounding_check: str | None = None
    cited: list[SourceRef] = Field(default_factory=list)  # sources still cited after the check
    final_answer: str | None = None
    refusal_reason: str | None = None
    notes: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Limits(BaseModel):
    max_steps: int
    max_model_calls: int
    max_tool_calls: int
    max_tokens: int
    timeout_s: float
    max_cost_usd: float | None = None  # None while we only use free tiers


# Section 4 of the design, per route. Before routing, the legal_question limits apply.
LIMITS: dict[str, Limits] = {
    "lookup": Limits(
        max_steps=6, max_model_calls=3, max_tool_calls=2, max_tokens=12_000, timeout_s=30
    ),
    "legal_question": Limits(
        max_steps=14, max_model_calls=9, max_tool_calls=6, max_tokens=40_000, timeout_s=90
    ),
    "compare": Limits(
        max_steps=14, max_model_calls=9, max_tool_calls=6, max_tokens=40_000, timeout_s=90
    ),
    "compliance_check": Limits(
        max_steps=30, max_model_calls=16, max_tool_calls=15, max_tokens=120_000, timeout_s=180
    ),
    "out_of_scope": Limits(
        max_steps=6, max_model_calls=3, max_tool_calls=0, max_tokens=12_000, timeout_s=30
    ),
}
MAX_SEARCHES = 3  # the first search and at most 2 re-searches
MAX_CRITIC_ROUNDS = 2
MAX_STEP_TOOL_CALLS = 3
MAX_TURNS = 6


class BudgetUsed(BaseModel):
    steps: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    unreported_token_calls: int = 0  # calls whose provider gave no token counts
    estimated_tokens: int = 0  # 4 characters per token, for those calls only
    started_at: float = Field(default_factory=time.time)
    elapsed_s: float = 0.0
    cost_usd: float | None = None  # only with a sourced price file (CLAUDE.md rule)
    models: list[str] = Field(default_factory=list)  # "provider:model" per call

    def tokens(self) -> int:
        """Tokens counted against the limit: estimates only once more than 2 calls were unknown."""
        estimate = self.estimated_tokens if self.unreported_token_calls > 2 else 0
        return self.input_tokens + self.output_tokens + estimate


class TraceEvent(BaseModel):
    type: str  # "node", "route", "plan", "tool_call", "tool_result", "step", "limit", ...
    node: str
    at_ms: float
    data: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    conversation: ConversationState = Field(default_factory=ConversationState)
    task: TaskState = Field(default_factory=TaskState)
    limits: Limits = Field(default_factory=lambda: LIMITS["legal_question"])
    used: BudgetUsed = Field(default_factory=BudgetUsed)
    status: Status = "running"
    limit_hit: LimitName | None = None
    trace: list[TraceEvent] = Field(default_factory=list)


STATE_TYPES = (
    Turn, ConversationState, SourceRef, SearchAttempt, StepResult, ToolCall, TaskState, Limits,
    BudgetUsed, TraceEvent, AgentState,
)  # fmt: skip
