"""Budget limits, the critic's reserve, token estimates, and repeated-call (loop) detection."""

import time

import pytest

from agent_support import tools_for
from copilot.agent.budget import call_key, limit_hit, record_model_calls
from copilot.agent.context import BudgetExceeded, LoopStop, Models, RunContext, Scope
from copilot.agent.state import AgentState, BudgetUsed, Limits, TaskState
from copilot.llm.base import Usage

LIMITS = Limits(max_steps=10, max_model_calls=3, max_tool_calls=2, max_tokens=10_000, timeout_s=30)


def state(**used) -> AgentState:
    return AgentState(task=TaskState(question="q", language="en"), limits=LIMITS,
                      used=BudgetUsed(**used))  # fmt: skip


# --- limit_hit --------------------------------------------------------------------------------


def test_nothing_hit_on_a_fresh_run():
    assert limit_hit(state(), "router") is None


def test_steps_limit():
    assert limit_hit(state(steps=10), "router") == "steps"


def test_model_calls_keep_one_back_for_the_critic():
    assert limit_hit(state(model_calls=2), "answer") == "model_calls"  # 3 minus the reserve
    assert limit_hit(state(model_calls=2), "ground") is None  # the critic may use it
    assert limit_hit(state(model_calls=3), "ground") == "model_calls"


def test_tokens_keep_6000_back_for_the_critic():
    assert limit_hit(state(input_tokens=4_100), "answer") == "tokens"  # 10,000 - 6,000
    assert limit_hit(state(input_tokens=4_100), "ground") is None


def test_unreported_tokens_are_estimated_only_after_two_unknown_calls():
    two = state(unreported_token_calls=2, estimated_tokens=50_000)
    assert limit_hit(two, "ground") is None
    three = state(unreported_token_calls=3, estimated_tokens=50_000)
    assert limit_hit(three, "ground") == "tokens"


def test_timeout():
    assert limit_hit(state(started_at=time.time() - 31), "router") == "timeout"


def test_cost_only_with_a_ceiling():
    assert limit_hit(state(cost_usd=99.0), "router") is None  # no ceiling while on free tiers
    capped = state(cost_usd=0.02).model_copy(
        update={"limits": LIMITS.model_copy(update={"max_cost_usd": 0.01})}
    )
    assert limit_hit(capped, "router") == "cost"


def test_record_model_calls_counts_successes_and_unknown_tokens():
    class LLM:
        usage = [
            Usage("gemini", "", "generate", None, None, 1.0, "503"),  # a failed attempt
            Usage("gemini", "m1", "generate", 100, 20, 1.0),
            Usage("groq", "m2", "generate", None, None, 1.0),
        ]

    used = record_model_calls(BudgetUsed(), LLM(), prompt_chars=400, output_chars=0)
    assert used.model_calls == 2 and used.models == ["gemini:m1", "groq:m2"]
    assert (used.input_tokens, used.output_tokens) == (100, 20)
    assert used.unreported_token_calls == 1 and used.estimated_tokens == 100


# --- repeated calls and the tool budget ------------------------------------------------------


def test_call_key_normalizes_case_whitespace_and_order():
    a = call_key("search_regulation", {"query": "Prohibited  Practices", "language": "en"})
    b = call_key("search_regulation", {"language": "en", "query": "prohibited practices"})
    assert a == b


@pytest.fixture
def scope_for():
    tools = tools_for()

    def make(node: str, limits: Limits = LIMITS) -> Scope:
        s = AgentState(task=TaskState(question="q", language="en"), limits=limits)
        return Scope(s, RunContext(tools, Models({})), node)

    return make


ARGS = {"regulation": "gdpr", "reference": "22", "language": "en"}


def test_second_identical_call_returns_the_earlier_result(scope_for):
    scope = scope_for("search", LIMITS.model_copy(update={"max_tool_calls": 5}))
    first = scope.tool("get_article", ARGS)
    second = scope.tool("get_article", dict(ARGS, reference=" 22 "))  # same after normalizing
    assert first.ok and second.error.error == "repeated_call"
    assert second.output == first.output and "already called" in second.error.message
    with pytest.raises(LoopStop):
        scope.tool("get_article", ARGS)


def test_tool_budget_stops_the_node(scope_for):
    scope = scope_for("search", LIMITS.model_copy(update={"max_tool_calls": 1}))
    scope.tool("get_article", ARGS)
    with pytest.raises(BudgetExceeded) as hit:
        scope.tool("get_article", dict(ARGS, reference="21"))
    assert hit.value.limit == "tool_calls"


def test_tools_outside_the_nodes_allow_list_are_refused(scope_for):
    scope = scope_for("lookup")
    result = scope.tool("search_regulation", {"query": "x y z", "language": "en"})
    assert result.error.error == "not_allowed" and "get_article" in result.error.message
    assert scope.used.tool_calls == 1  # refused calls still count toward the limit


def test_gdpr_overlap_step_may_only_search_the_gdpr(scope_for):
    scope = scope_for("step_gdpr_overlap")
    result = scope.tool("search_regulation", {"query": "impact assessment", "regulation": "aiact",
                                              "language": "en"})  # fmt: skip
    assert result.error.error == "invalid_argument" and '"gdpr"' in result.error.message


def test_invalid_arguments_count_toward_the_tool_limit(scope_for):
    scope = scope_for("search")
    result = scope.tool("get_article", {"regulation": "gdpr", "language": "en"})
    assert result.error.error == "invalid_argument" and scope.used.tool_calls == 1


def test_model_calls_go_through_the_budget(scope_for):
    from agent_support import FakeLLM

    log = []
    scope = scope_for("router")
    scope.ctx.models = Models({"router": lambda: FakeLLM(lambda p: "ok", log)})
    for _ in range(2):
        scope.model("router", "prompt")
    with pytest.raises(BudgetExceeded):
        scope.model("router", "prompt")  # the third call would eat the critic's reserve
    assert len(log) == 2 and scope.used.input_tokens == 200
