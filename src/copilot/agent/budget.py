"""Budget guard and loop detection (docs/agent-design.md, section 4)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from copilot.agent.state import AgentState, BudgetUsed, LimitName

# Kept back for the grounding check, so a draft can still be verified when everything else is
# spent: one model call and 6,000 tokens.
CRITIC_RESERVE_CALLS = 1
CRITIC_RESERVE_TOKENS = 6_000
CRITIC_NODES = {"ground", "finalize_partial"}
PRICES = Path("evals/prices.json")


def limit_hit(state: AgentState, node: str) -> LimitName | None:
    """The first limit the run has reached before entering `node`, or None.

    Nodes other than the critic see the limits minus the critic's reserve.
    """
    used, limits = state.used, state.limits
    reserve = node not in CRITIC_NODES
    calls_left = limits.max_model_calls - (CRITIC_RESERVE_CALLS if reserve else 0)
    tokens_left = limits.max_tokens - (CRITIC_RESERVE_TOKENS if reserve else 0)
    if node != "finalize_partial" and used.steps >= limits.max_steps:
        return "steps"  # the last check of a partial draft is not a new step of work
    if time.time() - used.started_at >= limits.timeout_s:
        return "timeout"
    if used.model_calls >= calls_left:
        return "model_calls"
    if used.tokens() >= tokens_left:
        return "tokens"
    if used.tool_calls > limits.max_tool_calls:
        return "tool_calls"
    if limits.max_cost_usd is not None and (used.cost_usd or 0.0) >= limits.max_cost_usd:
        return "cost"
    return None


def record_model_calls(used: BudgetUsed, llm, prompt_chars: int, output_chars: int) -> BudgetUsed:
    """Add the usage records of one FallbackLLM (one per logical call) to the budget."""
    new = used.model_copy(deep=True)
    for record in getattr(llm, "usage", []):
        if record.error is not None:
            continue
        new.model_calls += 1
        new.models.append(f"{record.provider}:{record.model}")
        if record.input_tokens is None or record.output_tokens is None:
            new.unreported_token_calls += 1
            new.estimated_tokens += (prompt_chars + output_chars) // 4
        else:
            new.input_tokens += record.input_tokens
            new.output_tokens += record.output_tokens
            new.cost_usd = _add_cost(new.cost_usd, record)
    new.elapsed_s = time.time() - new.started_at
    return new


def _add_cost(total: float | None, record) -> float | None:
    """Dollars only from a sourced price file; otherwise stays None (never invented)."""
    if not PRICES.exists():
        return total
    price = json.loads(PRICES.read_text("utf-8")).get(record.model)
    if not price:
        return total
    cost = (
        record.input_tokens / 1e6 * price["input_per_1m_tokens_usd"]
        + record.output_tokens / 1e6 * price["output_per_1m_tokens_usd"]
    )
    return (total or 0.0) + cost


def call_key(name: str, args: dict) -> str:
    """Tool name plus normalized arguments: lowercased, whitespace folded, keys sorted."""

    def norm(value):
        if isinstance(value, str):
            return " ".join(value.lower().split())
        if isinstance(value, dict):
            return {k: norm(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [norm(v) for v in value]
        return value

    return name + ":" + json.dumps(norm(args), sort_keys=True, ensure_ascii=False)


def repeat_count(state: AgentState, key: str) -> int:
    """How many times this exact call was already made in this run."""
    return sum(1 for call in state.task.tool_calls if call.key == key)
