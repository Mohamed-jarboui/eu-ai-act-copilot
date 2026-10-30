"""classify_ai_risk: the model extracts facts, rules decide (owner decision, Stage 4A).

The small model only fills a fixed checklist from the system description. The category comes
from the deterministic rules below, in a fixed order, and every rule records the provision it
rests on. A deciding fact the description does not settle becomes a question for the user.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

Value = Literal["yes", "no", "unknown"]
Role = Literal["provider", "deployer", "importer", "distributor", "authorised_representative"]
Category = Literal["prohibited", "high_risk", "limited", "minimal", "undetermined"]
RULES_PATH = Path(__file__).parent / "data" / "risk_rules.json"
ANNEX_III_AREAS = {str(n) for n in range(1, 9)}


@cache
def rules() -> dict:
    return json.loads(RULES_PATH.read_text("utf-8"))


def fact_names() -> list[str]:
    r = rules()
    return [
        name for group in ("prohibited", "high_risk", "transparency", "gpai") for name in r[group]
    ]


class Fact(BaseModel):
    value: Value = "unknown"
    evidence: str = ""  # the phrase of the description that supports the value


class RiskFacts(BaseModel):
    facts: dict[str, Fact] = Field(default_factory=dict)
    annex_iii_area: str = "unknown"  # "1".."8", "none" or "unknown"
    roles: list[Role] = Field(default_factory=list)  # empty = unknown

    def get(self, name: str) -> Value:
        return self.facts.get(name, Fact()).value


class RiskResult(BaseModel):
    category: Category
    transparency_obligations: bool
    gpai: bool
    basis: list[str]  # cite ids the decision rests on
    annex_iii_point: str | None = None
    roles: list[Role] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)  # questions, in the user's language
    assumptions: list[str] = Field(default_factory=list)
    derogation_considered: bool = False


def extraction_prompt(description: str, known: RiskFacts | None) -> str:
    r = rules()
    lines = []
    for group in ("prohibited", "high_risk", "transparency", "gpai"):
        for name, meta in r[group].items():
            lines.append(f'- "{name}": the system {meta["fact"]}')
    known_block = (
        f"\nFacts already confirmed by the user (keep them):\n{known.model_dump_json()}\n"
        if known
        else ""
    )
    return (
        "Fill a fact checklist about an AI system from its description. Use ONLY the "
        "description: do not infer beyond it.\n"
        'Value rules: "yes" if the description says so; "no" if nothing in the description '
        'suggests it; "unknown" only if the description makes it plausible but does not say. '
        "For each value give the phrase of the description that supports it (empty for no).\n\n"
        "Facts:\n" + "\n".join(lines) + "\n\n"
        '"annex_iii_area" is the area number "1" to "8" described in the fact above, "none", '
        'or "unknown".\n"roles": the roles the user plays among provider (develops the '
        "system or has it developed and puts it on the market under its name), deployer (uses "
        "it under its authority), importer, distributor, authorised_representative; empty list "
        f"if the description does not say.\n{known_block}\n"
        f"Description:\n<<<\n{description}\n>>>\n\n"
        'Reply with ONLY JSON: {"facts": {"<fact name>": {"value": "yes|no|unknown", '
        '"evidence": "..."}}, "annex_iii_area": "...", "roles": [...]}'
    )


def parse_facts(text: str) -> RiskFacts | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        facts = RiskFacts.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None
    known = set(fact_names())
    facts.facts = {k: v for k, v in facts.facts.items() if k in known}
    if facts.annex_iii_area not in ANNEX_III_AREAS | {"none", "unknown"}:
        facts.annex_iii_area = "unknown"
    return facts


def merge(extracted: RiskFacts, known: RiskFacts | None) -> RiskFacts:
    """Facts confirmed by the user win over extracted ones."""
    if not known:
        return extracted
    merged = extracted.model_copy(deep=True)
    merged.facts.update({k: v for k, v in known.facts.items() if v.value != "unknown"})
    if known.annex_iii_area != "unknown":
        merged.annex_iii_area = known.annex_iii_area
    merged.roles = known.roles or extracted.roles
    return merged


def decide(facts: RiskFacts, language: str) -> RiskResult:
    """The deterministic rules, in the order of docs/agent-design.md (section 3)."""
    r = rules()
    basis: list[str] = []
    missing: list[str] = []
    assumptions: list[str] = []

    def ask(group: str, name: str) -> None:
        missing.append(r[group][name]["question"][language])

    # 1. prohibited practices, Art. 5(1)(a) to (h)
    prohibited = [n for n in r["prohibited"] if facts.get(n) == "yes"]
    for name in r["prohibited"]:
        if facts.get(name) == "unknown":
            ask("prohibited", name)
    # transparency (Art. 50) and GPAI are reported whatever the category
    transparency = [n for n in r["transparency"] if facts.get(n) == "yes"]
    for name in r["transparency"]:
        if facts.get(name) == "unknown":
            ask("transparency", name)
    gpai = facts.get("gpai_model") == "yes"
    if facts.get("gpai_model") == "unknown":
        ask("gpai", "gpai_model")
    if not facts.roles:
        missing.append(r["role_question"][language])

    def result(category: Category, **extra) -> RiskResult:
        extra_basis = [r["transparency"][n]["basis"] for n in transparency]
        extra_basis += [r["gpai"]["gpai_model"]["basis"]] if gpai else []
        return RiskResult(
            category=category,
            transparency_obligations=bool(transparency),
            gpai=gpai,
            basis=list(dict.fromkeys(basis + extra_basis)),
            roles=facts.roles,
            missing_facts=missing,
            assumptions=assumptions,
            **extra,
        )

    if prohibited:
        basis += [r["prohibited"][n]["basis"] for n in prohibited]
        return result("prohibited")

    # 2. safety component under Annex I legislation, Art. 6(1)
    hr = r["high_risk"]
    if facts.get("annex_i_safety_component") == "yes":
        basis.append(hr["annex_i_safety_component"]["basis"])
        return result("high_risk")
    if facts.get("annex_i_safety_component") == "unknown":
        ask("high_risk", "annex_i_safety_component")

    # 3. Annex III use case, Art. 6(2), with the Art. 6(3) derogation and its profiling exclusion
    area = facts.annex_iii_area
    if area == "unknown":
        ask("high_risk", "annex_iii_area")
        return result("undetermined")
    if area in ANNEX_III_AREAS:
        point = f"aiact:anx:III:{area}"
        derogation = facts.get("art6_3_derogation")
        profiling = facts.get("profiling")
        if derogation == "yes" and profiling == "no":
            basis += [hr["art6_3_derogation"]["basis"], point]
            assumptions.append(_text("derogation_applies", language))
            category: Category = "limited" if transparency else "minimal"
            return result(category, annex_iii_point=point, derogation_considered=True)
        basis += [hr["annex_iii_area"]["basis"], point]
        if profiling == "yes":
            basis.append(hr["profiling"]["basis"])
        elif derogation == "unknown":
            ask("high_risk", "art6_3_derogation")
            assumptions.append(_text("derogation_assumed_not_met", language))
        if profiling == "unknown" and derogation == "yes":
            ask("high_risk", "profiling")
        return result("high_risk", annex_iii_point=point, derogation_considered=True)

    # 4. no high-risk ground: limited if a transparency duty applies, else minimal
    return result("limited" if transparency else "minimal")


_TEXTS = {
    "derogation_applies": {
        "en": "Art. 6(3) derogation assumed to apply as described; the provider must document "
        "this assessment before placing the system on the market.",
        "fr": "Dérogation de l'art. 6, paragraphe 3, supposée applicable telle que décrite ; le "
        "fournisseur doit documenter cette évaluation avant la mise sur le marché.",
        "de": "Ausnahme nach Art. 6 Abs. 3 als gegeben angenommen; der Anbieter muss diese "
        "Bewertung vor dem Inverkehrbringen dokumentieren.",
    },
    "derogation_assumed_not_met": {
        "en": "Assumed that the Art. 6(3) derogation does not apply, because the description "
        "does not say so.",
        "fr": "Supposé que la dérogation de l'art. 6, paragraphe 3, ne s'applique pas, la "
        "description ne l'indiquant pas.",
        "de": "Angenommen, dass die Ausnahme nach Art. 6 Abs. 3 nicht greift, da die "
        "Beschreibung dies nicht angibt.",
    },
}


def _text(key: str, language: str) -> str:
    return _TEXTS[key].get(language, _TEXTS[key]["en"])
