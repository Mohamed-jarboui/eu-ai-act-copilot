"""`python -m copilot.agent.traces`: run one real question per graph path and write the traces
to docs/traces/ (raw JSON and a readable Markdown version). Needs the API keys."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from copilot.agent.factory import build_agent
from copilot.agent.guards import redact

OUT = Path("docs/traces")
RUNS = [
    ("legal_question", "en", "Does a chatbot have to tell people they are talking to a machine?"),
    ("compliance_check", "en", "We are building a CV-screening tool that ranks job applicants "
     "for our clients. What applies to us under the AI Act?"),
    ("lookup", "fr", "Que dit l'article 22 du RGPD ?"),
]  # fmt: skip


def describe(event: dict) -> str | None:
    """One line per trace event, as in the chat page's reasoning panel."""
    d, kind = event.get("data", {}), event["type"]
    if kind == "route":
        extra = f" -> {d['reference']}" if d.get("reference") else ""
        return f"Route: **{d['route']}** (confidence {d['confidence']}, {d['method']}){extra}"
    if kind == "plan":
        return "Plan: " + " -> ".join(d["steps"])
    if kind == "rewrite":
        return "Search queries: " + "; ".join(q.split("|", 1)[1] for q in d["queries"])
    if kind == "tool_call":
        return f"Tool `{d['tool']}` {json.dumps(d['args'], ensure_ascii=False)[:200]}"
    if kind == "tool_result":
        if d.get("error"):
            return f"  -> error `{d['error']}`: {d['message']}"
        return f"  -> {d.get('summary') or 'ok'} ({d['ms']} ms)"
    if kind == "grade":
        return f"Grade: **{d['decision']}**" + (f", relevant: {', '.join(d['relevant'])}"
                                              if d.get("relevant") else "")  # fmt: skip
    if kind == "model_call":
        failed = f"; {len(d['failed'])} failed attempt(s) first" if d.get("failed") else ""
        return f"Model for {d['purpose']}: {', '.join(d['models'])} ({d['ms']} ms{failed})"
    if kind == "guard":
        return f"Guard: {d['rule']}"
    if kind == "step":
        return f"Step **{d['step']}**: {d['status']}. {d['summary']}"
    if kind == "draft":
        return f"Draft: {d['words']} words from {', '.join(d['sources'][:8])}"
    if kind == "ground":
        return f"Grounding check: kept {d['kept']}, removed {d['removed']} (round {d['round']})"
    if kind in ("limit", "loop", "partial", "error"):
        return f"**{kind}**: {json.dumps(d, ensure_ascii=False)}"
    return None


def markdown(route: str, question: str, final: dict, events: list[dict]) -> str:
    steps = [line for e in events if (line := describe(e))]
    u = final["usage"]
    cites = "\n".join(f"- {c['label']}: {c['snippet'][:160]}" for c in final["citations"]) or "none"
    return "\n".join([
        f"# Trace: {route}",
        "",
        f"Captured {date.today().isoformat()} with `python -m copilot.agent.traces` (real models, "
        "unedited). Raw events: the `.json` file next to this one.",
        "",
        f"**Question ({final['language']}):** {question}",
        "",
        f"**Outcome:** route `{final['route']}`, status `{final['status']}`, grounded "
        f"`{final['grounded']}`, retrieval `{final['retrieval_mode']}`, limit hit "
        f"`{final['limit_hit']}`.",
        "",
        f"**Usage:** {u['elapsed_s']:.1f} s, {u['steps']} steps, {u['model_calls']} model calls "
        f"({', '.join(sorted(set(u['models'])))}), {u['tool_calls']} tool calls, "
        f"{u['input_tokens']} input / {u['output_tokens']} output tokens.",
        "",
        "## Reasoning steps",
        "",
        *[f"{i}. {line}" for i, line in enumerate(steps, start=1)],
        "",
        "## Answer",
        "",
        *[f"> {line}" if line else ">" for line in final["answer"].split("\n")],
        "",
        "## Citations",
        "",
        cites,
        "",
    ] + ([f"Notes: {', '.join(final['notes'])}", ""] if final["notes"] else [])
      + ([f"Retrieval notes: {'; '.join(final['retrieval_notes'])}", ""]
         if final["retrieval_notes"] else []))  # fmt: skip


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    agent = build_agent()
    for route, language, question in RUNS:
        print(f"== {route}: {question}", flush=True)
        final, events = agent.run(question, language)
        data = final.model_dump()
        raw = {"question": redact(question), "final": data, "events": events}
        (OUT / f"{route}.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n",
                                           "utf-8")  # fmt: skip
        (OUT / f"{route}.md").write_text(markdown(route, question, data, events), "utf-8")
        print(f"   route={final.route} status={final.status} grounded={final.grounded} "
              f"{final.usage['elapsed_s']:.1f}s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
