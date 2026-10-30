"""The router: an explicit-reference check without a model, then one small-model call."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from copilot.answer.citations import parse_citation

ROUTES = ("lookup", "legal_question", "compliance_check", "compare", "out_of_scope")
CONFIDENCE_THRESHOLD = 0.6  # placeholder until calibrated on route-labelled questions (design 9)

# Names of the two regulations in EN/FR/DE, mapped to our ids.
_REG_NAMES = {
    "aiact": r"AI[ -]?Act|AI Regulation|KI[- ]?(?:Verordnung|VO)|"
    r"règlement (?:sur l['’]|européen sur l['’])IA|RIA",
    "gdpr": r"GDPR|RGPD|DSGVO|DS-GVO|General Data Protection Regulation",
}
_REF = (
    r"(?:Art(?:icle|ikel)?\.?|art(?:icle)?\.?)\s*(?P<art>\d+[a-z]?)(?P<paras>(?:\s*\(\s*\w+\s*\))*)"
    r"(?:\s*(?:,\s*)?(?:paragraphe|paragraph|Absatz|Abs\.)\s*(?P<para>\d+))?"
)
_ANNEX = (
    r"(?:Annex(?:e)?|Anhang)\s+(?P<annex>[IVXLC]+)\b"
    r"(?:\s*,?\s*(?:point|Nummer|Nr\.)\s*(?P<point>\d+))?"
)
# Wording that asks for the provision itself rather than a question about a situation.
_LOOKUP_ASK = (
    r"^(?:what (?:does|do)|show|print|quote|give me|text of|read|explain)|\b(?:say|says|text)\b|"
    r"^(?:que dit|que disent|montre|affiche|donne|texte de|explique)|"
    r"^(?:was (?:steht|sagt|regelt)|zeige|zitiere|gib mir|erkläre)|wortlaut"
)


@dataclass
class RouteDecision:
    route: str
    confidence: float
    regulation: str | None = None  # "aiact" | "gdpr" | "both"
    reference: str | None = None  # cite id for lookup
    clarifying_question: str | None = None
    method: str = "rules"  # "rules" or "model"
    clarify: bool = False


def find_reference(question: str) -> tuple[str, str] | None:
    """(regulation, cite id) when the question names exactly one provision of one regulation."""
    for regulation, names in _REG_NAMES.items():
        if not re.search(names, question, re.IGNORECASE):
            continue
        name = "AI Act" if regulation == "aiact" else "GDPR"
        if m := re.search(_ANNEX, question):
            label = f"{name} Annex {m.group('annex')}" + (
                f", point {m.group('point')}" if m.group("point") else ""
            )
            if cid := parse_citation(label):
                return regulation, cid
        if m := re.search(_REF, question, re.IGNORECASE):
            paras = m.group("paras") or (f"({m.group('para')})" if m.group("para") else "")
            if cid := parse_citation(f"{name} Art. {m.group('art')}{paras}"):
                return regulation, cid
    return None


def rule_route(question: str) -> RouteDecision | None:
    """Route without a model when the question asks for a named provision."""
    found = find_reference(question)
    if not found:
        return None
    other = re.findall(r"\b(?:Art(?:icle|ikel)?\.?\s*\d+|Annex(?:e)?\s+[IVX]+|Anhang\s+[IVX]+)",
                       question, re.IGNORECASE)  # fmt: skip
    short = len(question.split()) <= 12
    if len(other) <= 1 and (short or re.search(_LOOKUP_ASK, question.strip(), re.IGNORECASE)):
        regulation, cid = found
        return RouteDecision("lookup", 1.0, regulation=regulation, reference=cid)
    return None


def router_prompt(question: str, turns: list[str]) -> str:
    history = "\n".join(turns[-4:]) or "(none)"
    return (
        "Classify a request to an assistant that answers ONLY about the EU AI Act (Regulation "
        "(EU) 2024/1689) and the GDPR (Regulation (EU) 2016/679).\n\n"
        "Routes:\n"
        '- "lookup": the user asks for the text or meaning of one named provision ("What does '
        'Article 22 GDPR say?").\n'
        '- "legal_question": a question about what the AI Act or the GDPR require, allow or '
        "define, including questions that need both regulations.\n"
        '- "compliance_check": the user describes their own AI system, product or use case and '
        'asks what applies to it ("We are building a CV-screening tool, what do we need to do?").\n'
        '- "compare": the user explicitly asks how the AI Act and the GDPR differ or relate on a '
        "topic.\n"
        '- "out_of_scope": anything not about these two regulations (other laws, tax, '
        "technology, general knowledge).\n\n"
        "Give a confidence between 0 and 1. If the request is too vague to route (for example "
        "'what applies to me?' without saying anything about the system), give a low confidence "
        "and write ONE short clarifying question in the user's language.\n\n"
        f"Recent conversation:\n{history}\n\nRequest:\n<<<\n{question}\n>>>\n\n"
        'Reply with ONLY JSON: {"route": "...", "confidence": 0.0, "regulation": '
        '"aiact|gdpr|both", "clarifying_question": null}'
    )


def parse_route(text: str) -> RouteDecision | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    route = data.get("route")
    if route not in ROUTES:
        return None
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    regulation = (
        data.get("regulation") if data.get("regulation") in ("aiact", "gdpr", "both") else None
    )
    question = data.get("clarifying_question") or None
    return RouteDecision(route, confidence, regulation=regulation, clarifying_question=question,
                         method="model")  # fmt: skip


def settle(decision: RouteDecision, question: str, already_asked: bool) -> RouteDecision:
    """Apply the routing policy to a model decision; used by the router node and evals.

    * A "lookup" needs a named provision: if the model chose lookup but the question names none,
      it is a legal question (measured: the model sent "Can I get a copy of my data?" to lookup).
    * Low confidence asks one clarifying question, whatever the route (design section 1). An
      earlier rule that never clarified out-of-scope routes made all 3 vague test questions
      refusals, because the model labels them out of scope with confidence 0.1 to 0.2.
    * At most one clarification per question: after one, the best route is used.
    """
    if decision.route == "lookup" and not decision.reference:
        found = find_reference(question)
        if found:
            decision.regulation, decision.reference = found
        else:
            decision.route = "legal_question"
    decision.clarify = decision.confidence < CONFIDENCE_THRESHOLD and not already_asked
    return decision
