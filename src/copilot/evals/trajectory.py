"""Rules over a run's trace events (docs/design.md, evaluation table: "Trajectory").

Each check returns True (passed), False (failed) or None (does not apply to this run), so the
report can give every rate with its own sample size.
"""

from __future__ import annotations

from copilot.agent.budget import call_key
from copilot.agent.state import MAX_SEARCHES

RETRIEVAL_TOOLS = {
    "search_regulation", "get_article", "get_definitions", "cross_reference", "list_obligations",
    "classify_ai_risk",
}  # fmt: skip
PLAN_STEPS = ("classify", "obligations", "gdpr_overlap")
# A tool call is "wrong" when the model chose badly; no_results, insufficient_facts and budget
# are normal outcomes of a correct call.
WRONG_CALL_ERRORS = {"invalid_argument", "not_found", "not_allowed", "repeated_call", "too_long"}


def _route(events: list[dict]) -> str | None:
    return next((e["data"]["route"] for e in events if e["type"] == "route"), None)


def retrieved_before_answering(events: list[dict]) -> bool | None:
    """Every draft is written after at least one successful retrieval tool call."""
    retrieved = False
    drafted = False
    for e in events:
        if e["type"] == "tool_result" and e["data"].get("ok") and \
                e["data"].get("tool") in RETRIEVAL_TOOLS:  # fmt: skip
            retrieved = True
        if e["type"] == "draft":
            drafted = True
            if not retrieved:
                return False
    return True if drafted else None


def search_rounds(events: list[dict]) -> int:
    return sum(1 for e in events if e["type"] == "node" and e["data"].get("name") == "search")


def within_search_limit(events: list[dict], limit: int = MAX_SEARCHES) -> bool | None:
    rounds = search_rounds(events)
    return rounds <= limit if rounds else None


def plan_complete(events: list[dict]) -> bool | None:
    """A compliance run that made a plan reports a result for every plan step."""
    if _route(events) != "compliance_check" or not any(e["type"] == "plan" for e in events):
        return None
    done = {e["data"].get("step") for e in events if e["type"] == "step"}
    return all(step in done for step in PLAN_STEPS)


def no_repeated_calls(events: list[dict]) -> bool | None:
    keys = [call_key(e["data"]["tool"], e["data"].get("args", {}))
            for e in events if e["type"] == "tool_call"]  # fmt: skip
    if any(e["type"] == "loop" for e in events):  # a third identical call stopped the run
        return False
    return len(keys) == len(set(keys)) if keys else None


def check(events: list[dict]) -> dict[str, bool | None]:
    return {
        "retrieved_before_answering": retrieved_before_answering(events),
        "within_search_limit": within_search_limit(events),
        "plan_complete": plan_complete(events),
        "no_repeated_calls": no_repeated_calls(events),
    }


def tool_calls(events: list[dict]) -> tuple[int, int]:
    """(tool calls made, calls whose error shows a wrong choice of tool or arguments)."""
    results = [e["data"] for e in events if e["type"] == "tool_result"]
    wrong = sum(1 for r in results if r.get("error") in WRONG_CALL_ERRORS or r.get("repeated"))
    return len(results), wrong
