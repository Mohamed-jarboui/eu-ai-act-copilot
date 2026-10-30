"""The six read-only tools (docs/agent-design.md, section 3) and the runner that guards them.

Every tool takes Pydantic-validated arguments, returns bounded output, and on failure returns a
ToolError whose message tells the model how to recover. Tools that return legal text also hand
back SourceRefs, so whatever the answer cites can be checked by the Stage 3 citation validator.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from copilot.agent import risk
from copilot.agent.state import SourceRef
from copilot.answer.citations import parse_citation
from copilot.answer.labels import cite_id, label_for
from copilot.ingest.models import Unit
from copilot.ingest.parse import DEFINITIONS_ARTICLE
from copilot.retrieve.tokenize import fold

DATA = Path(__file__).parent / "data"
Language = Literal["en", "fr", "de"]
Regulation = Literal["aiact", "gdpr"]
SNIPPET_CHARS, ARTICLE_CHARS, DEFINITION_CHARS, QUOTE_CHARS = 600, 4000, 1500, 300
SOURCE_CHARS = 6000
ARTICLE_RANGE = {"aiact": 113, "gdpr": 99}
ANNEXES = {"aiact": 13, "gdpr": 0}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]


class ToolError(BaseModel):
    error: Literal[
        "invalid_argument",
        "not_found",
        "no_results",
        "too_long",
        "insufficient_facts",
        "budget",
        "repeated_call",
        "not_allowed",
    ]
    message: str  # written for the model: what went wrong and what to do instead


# --- argument schemas ----------------------------------------------------------------------


class SearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    regulation: Literal["aiact", "gdpr", "both"] = "both"
    language: Language
    k: int = Field(default=6, ge=1, le=8)


class ArticleArgs(BaseModel):
    regulation: Regulation
    reference: str = Field(min_length=1, max_length=40)
    language: Language


class DefinitionArgs(BaseModel):
    term: str = Field(min_length=2, max_length=80)
    regulation: Literal["aiact", "gdpr", "both"] = "both"
    language: Language


class ClassifyArgs(BaseModel):
    system_description: str = Field(min_length=30)  # the upper bound is a too_long error
    known_facts: risk.RiskFacts | None = None
    language: Language


class ObligationArgs(BaseModel):
    role: Literal["provider", "deployer", "importer", "distributor", "authorised_representative"]
    category: Literal["high_risk", "limited", "gpai", "gpai_systemic"]
    language: Language


class CrossRefArgs(BaseModel):
    regulation: Regulation
    reference: str = Field(min_length=1, max_length=40)
    language: Language


# --- descriptions given to the model (section 3 of the design) -------------------------------

DESCRIPTIONS = {
    "search_regulation": (
        "Search the official text of the AI Act and the GDPR for passages relevant to a "
        "question. Use it when you need the provisions that answer a substantive question and "
        "you do not know the exact article. Write queries with the legal vocabulary of the "
        'regulation ("prohibited practices", not "banned AI"), in the language you were told to '
        "use. Do not use it to fetch an article whose number you already know (use get_article), "
        "nor to look up what a term means (use get_definitions). Results are passages, not "
        "answers: read them before relying on them."
    ),
    "get_article": (
        'Fetch the full text of one provision when you know its reference, for example "6(2)", '
        '"22" or "Annex III, point 4". Use it to read a passage that search returned only as a '
        "snippet, or when the user names a provision. Do not guess numbers: if you are not sure "
        "which article applies, search first."
    ),
    "get_definitions": (
        'Look up how the regulations define a term ("AI system", "provider", "personal data", '
        '"deep fake"). Use it before you rely on a defined term, or when the user asks what '
        "something means. Definitions come from AI Act Art. 3 and GDPR Art. 4 only; a term not "
        "defined there is not a legal definition, so do not use this tool for general vocabulary."
    ),
    "classify_ai_risk": (
        "Classify an AI system described by the user into the AI Act's categories: prohibited "
        "practice, high-risk, limited risk (transparency duties) or minimal risk. Use it only in "
        "a compliance check, once you have the system's purpose, users and context. It returns "
        "the category, the provisions it rests on, the facts it used, and the facts still "
        "missing. Never state a category without it, and pass its missing facts on to the user "
        "instead of filling them in yourself."
    ),
    "list_obligations": (
        "List the AI Act obligations that apply to one role for one risk category, as the "
        "provisions themselves (labels and short quotes), not as advice. Use it after "
        "classify_ai_risk, once per role the user plays (provider, deployer, importer, "
        "distributor, authorised_representative). Do not use it to decide the category, and do "
        "not paraphrase the result beyond what the quotes say."
    ),
    "cross_reference": (
        "Find provisions connected to a given one: the articles it refers to, the articles that "
        "refer to it, and the related provisions of the other regulation (AI Act <-> GDPR). Use "
        "it in comparisons and in the GDPR overlap step. Do not use it to search by topic."
    ),
}
ARG_MODELS: dict[str, type[BaseModel]] = {
    "search_regulation": SearchArgs,
    "get_article": ArticleArgs,
    "get_definitions": DefinitionArgs,
    "classify_ai_risk": ClassifyArgs,
    "list_obligations": ObligationArgs,
    "cross_reference": CrossRefArgs,
}


def tool_catalog(names: list[str]) -> str:
    """The tools a node may use, as shown to the model: name, description, argument schema."""
    blocks = []
    for name in names:
        schema = ARG_MODELS[name].model_json_schema()
        props = {
            k: {kk: vv for kk, vv in v.items() if kk != "title"}
            for k, v in schema.get("properties", {}).items()
        }
        blocks.append(
            f"- {name}: {DESCRIPTIONS[name]}\n  arguments: {json.dumps(props, ensure_ascii=False)}"
        )
    return "\n".join(blocks)


# --- the data the tools read ---------------------------------------------------------------


@dataclass
class ToolResult:
    output: dict[str, Any] | None = None
    error: ToolError | None = None
    sources: list[SourceRef] = field(default_factory=list)
    llm: Any = None  # the model wrapper used, so the runner can charge its usage

    @property
    def ok(self) -> bool:
        return self.error is None

    def for_model(self) -> str:
        """What the model sees: the output, or the error with its recovery message."""
        if self.error:
            return self.error.model_dump_json()
        return json.dumps(self.output, ensure_ascii=False)


def _err(kind: str, message: str) -> ToolResult:
    return ToolResult(error=ToolError(error=kind, message=message))


def _clip(text: str, limit: int) -> tuple[str, bool]:
    text = " ".join(text.split()) if limit <= SNIPPET_CHARS else text
    return (text, False) if len(text) <= limit else (text[:limit].rstrip() + "…", True)


class Tools:
    """The six tools over the corpus, the app retriever and the data tables."""

    def __init__(self, units: list[Unit], retriever, risk_llm_factory=None) -> None:
        self.units = {(u.language, u.id): u for u in units}
        self.retriever = retriever  # copilot.answer.retrieval.AppRetriever
        self.risk_llm_factory = risk_llm_factory  # small-tier model for fact extraction
        self.children: dict[tuple[str, str], list[Unit]] = defaultdict(list)
        for u in units:
            if u.parent_id:
                self.children[(u.language, u.parent_id)].append(u)

    # --- helpers ----------------------------------------------------------------------------

    def source(self, unit: Unit) -> SourceRef:
        cid = cite_id(unit)
        return SourceRef(
            cite_id=cid,
            label=label_for(cid),
            unit_id=unit.id,
            regulation=unit.regulation,
            title=unit.title,
            text=unit.text[:SOURCE_CHARS],
        )

    def resolve(self, regulation: str, reference: str, language: str) -> Unit | ToolResult:
        """'6(2)' / 'Article 6(2)' / 'Annex III, point 4' -> the corpus unit, or a ToolError."""
        ref = re.sub(r"^(?:article|art\.?|artikel)\s*", "", reference.strip(), flags=re.I)
        name = "AI Act" if regulation == "aiact" else "GDPR"
        label = (
            f"{name} {ref}"
            if ref.lower().startswith(("annex", "anhang", "annexe"))
            else (f"{name} Art. {ref}")
        )
        label = re.sub(r"(?i)\b(anhang|annexe)\b", "Annex", label)
        cid = parse_citation(label)
        if cid is None:
            return _err(
                "invalid_argument", "Write the reference like '6(2)' or 'Annex III, point 4'."
            )
        parts = cid.split(":")
        if parts[1] == "art":
            number = int(re.sub(r"\D", "", parts[2]) or 0)
            if not 1 <= number <= ARTICLE_RANGE[regulation]:
                return _err("not_found", self._range_message())
            if DEFINITIONS_ARTICLE[regulation] == parts[2] and len(parts) == 4:
                unit_id = f"{regulation}:def:{parts[3]}"  # Art. 3(1) is definition 1
            else:
                unit_id = cid
        else:
            if parts[2] not in ROMAN[: ANNEXES[regulation]]:
                return _err("not_found", self._range_message())
            unit_id = cid
        unit = self.units.get((language, unit_id))
        if unit is None and parts[1] == "art":  # second-list points ("l2") or unknown parts
            unit = next(
                (
                    u
                    for (lg, i), u in self.units.items()
                    if lg == language and re.sub(r":l\d+(?=:)", "", i) == unit_id
                ),
                None,
            )
        if unit is None:
            return _err(
                "not_found",
                f"{label_for(cid)} does not exist. Fetch the article without the paragraph to "
                "see which parts it has, or search.",
            )
        return unit

    @staticmethod
    def _range_message() -> str:
        return (
            "AI Act articles run from 1 to 113 and annexes I to XIII; GDPR articles from 1 to 99 "
            "and it has no annexes. Check the number, or search."
        )

    # --- the tools ----------------------------------------------------------------------------

    def search_regulation(self, a: SearchArgs) -> ToolResult:
        regulation = None if a.regulation == "both" else a.regulation
        found = self.retriever.retrieve(a.query, a.language, a.k, regulation=regulation)
        if not found.ids:
            return _err(
                "no_results",
                "No passage matched. Rephrase with the regulation's own terms, try the other "
                "regulation, or check the term with get_definitions. If two rephrasings fail, "
                "the texts probably do not cover this.",
            )
        items, sources = [], []
        for rank, chunk_id in enumerate(found.ids, start=1):
            unit = self.retriever.by_id[(a.language, chunk_id)]
            src = self.source(unit)
            snippet, _ = _clip(unit.text, SNIPPET_CHARS)
            items.append(
                {
                    "rank": rank,
                    "label": src.label,
                    "cite_id": src.cite_id,
                    "title": unit.title,
                    "snippet": snippet,
                }
            )
            sources.append(src)
        out = {
            "results": items,
            "mode": found.evidence.mode,
            "notes": found.notes,
            "evidence": found.evidence.__dict__,
        }
        return ToolResult(output=out, sources=sources)

    def get_article(self, a: ArticleArgs) -> ToolResult:
        unit = self.resolve(a.regulation, a.reference, a.language)
        if isinstance(unit, ToolResult):
            return unit
        text, truncated = _clip(unit.text, ARTICLE_CHARS)
        parts = [label_for(cite_id(c)) for c in self.children.get((a.language, unit.id), [])]
        src = self.source(unit)
        out = {
            "label": src.label,
            "cite_id": src.cite_id,
            "title": unit.title,
            "chapter": unit.chapter,
            "text": text,
            "truncated": truncated,
            "parts": parts[:40],
            "refs": list(unit.refs)[:20],
        }
        return ToolResult(output=out, sources=[src])

    def get_definitions(self, a: DefinitionArgs) -> ToolResult:
        regs = ["aiact", "gdpr"] if a.regulation == "both" else [a.regulation]
        defs = [
            u
            for (lg, _), u in self.units.items()
            if lg == a.language and u.type == "definition" and u.regulation in regs
        ]
        term = a.term.strip()

        def key(u: Unit) -> str:
            return fold(u.title or "")

        exact = [u for u in defs if (u.title or "").casefold() == term.casefold()]
        folded = [u for u in defs if key(u) == fold(term)]
        ranked = exact or folded
        if not ranked:
            scored = [(difflib.SequenceMatcher(None, key(u), fold(term)).ratio(), u) for u in defs]
            ranked = [u for score, u in sorted(scored, key=lambda s: -s[0]) if score >= 0.75]
        if not ranked:
            close = difflib.get_close_matches(fold(term), [key(u) for u in defs], n=5, cutoff=0.4)
            names = [u.title for u in defs if key(u) in close][:5]
            return _err(
                "no_results",
                f"No defined term matches. Closest defined terms: {', '.join(names) or 'none'}. "
                "If none fits, the term is not defined in these regulations.",
            )
        items, sources = [], []
        for unit in list(dict.fromkeys(ranked))[:3]:
            src = self.source(unit)
            text, _ = _clip(unit.text, DEFINITION_CHARS)
            items.append({"label": src.label, "term": unit.title, "text": text})
            sources.append(src)
        return ToolResult(output={"definitions": items}, sources=sources)

    def classify_ai_risk(self, a: ClassifyArgs) -> ToolResult:
        if len(a.system_description) > 3000:
            return _err(
                "too_long",
                "Shorten the description to 3000 characters: keep the "
                "purpose, the people affected, the users and the context.",
            )
        if self.risk_llm_factory is None:
            return _err("insufficient_facts", "The classifier is not available.")
        llm = self.risk_llm_factory()
        prompt = risk.extraction_prompt(a.system_description, a.known_facts)
        facts = None
        for _attempt in range(2):  # one retry: a reasoning model can cut its JSON short
            try:
                reply = llm.generate(prompt, tier="small", max_output_tokens=4000)
            except Exception as exc:  # provider down: tell the model, do not crash the run
                return ToolResult(
                    error=ToolError(
                        error="insufficient_facts",
                        message=f"Fact extraction failed ({type(exc).__name__}); try again once.",
                    ),
                    llm=llm,
                )
            facts = risk.parse_facts(reply.text)
            if facts is not None:
                break
        if facts is None:
            return ToolResult(
                error=ToolError(
                    error="insufficient_facts",
                    message="The fact extraction was unreadable twice. Finish this step and say "
                    "the system could not be classified; do not decide the category yourself.",
                ),
                llm=llm,
            )
        facts = risk.merge(facts, a.known_facts)
        said_anything = facts.annex_iii_area != "unknown" or any(
            f.value == "yes" for f in facts.facts.values()
        )
        if not said_anything and not facts.roles:
            return ToolResult(
                error=ToolError(
                    error="insufficient_facts",
                    message="The description does not say what the system is for or who uses "
                    "it. Ask the user for its purpose, who it affects, and where it is used, then "
                    "call again.",
                ),
                llm=llm,
            )
        result = risk.decide(facts, a.language)
        sources = []
        for cid in result.basis:
            unit = self._unit_for_cite(cid, a.language)
            if unit is not None:
                sources.append(self.source(unit))
        out = {
            **result.model_dump(),
            "basis": [label_for(c) for c in result.basis],
            "facts": {k: v.value for k, v in facts.facts.items()},
            "annex_iii_area": facts.annex_iii_area,
        }
        return ToolResult(output=out, sources=sources, llm=llm)

    def list_obligations(self, a: ObligationArgs) -> ToolResult:
        table = json.loads((DATA / "obligations.json").read_text("utf-8"))
        cids = table.get(f"{a.role}|{a.category}")
        if not cids:
            return _err(
                "not_found",
                "No obligations table entry for this role and category. Prohibited practices "
                "have no obligations: they are banned.",
            )
        items, sources = [], []
        for cid in cids:
            unit = self._unit_for_cite(cid, a.language)
            if unit is None:
                continue
            parts = self.children.get((a.language, unit.id)) or [unit]
            for part in parts:
                src = self.source(part)
                quote, _ = _clip(part.text, QUOTE_CHARS)
                items.append({"label": src.label, "title": unit.title, "quote": quote})
                sources.append(src)
        out = {"obligations": items[:20], "more": len(items) > 20}
        return ToolResult(output=out, sources=sources[:20])

    def cross_reference(self, a: CrossRefArgs) -> ToolResult:
        unit = self.resolve(a.regulation, a.reference, a.language)
        if isinstance(unit, ToolResult):
            return unit
        article = unit.article or (unit.id.split(":")[2] if ":art:" in unit.id else None)
        refs_to = []
        for ref in unit.refs:
            kind, num = ref.split(":")[:2]
            refs_to.append(
                self._label_title(
                    f"{a.regulation}:{'art' if kind == 'art' else 'anx'}:{num}", a.language
                )
            )
        referred = []
        if article:
            for (lg, _), u in self.units.items():
                if (
                    lg == a.language
                    and u.regulation == a.regulation
                    and u.type == "article"
                    and (f"art:{article}" in u.refs and u.article != article)
                ):
                    referred.append(
                        self._label_title(f"{a.regulation}:art:{u.article}", a.language)
                    )
        other = self._other_regulation(a.regulation, article, a.language) if article else []
        out = {
            "label": label_for(cite_id(unit)),
            "refers_to": _dedupe(refs_to)[:10],
            "referred_by": _dedupe(referred)[:10],
            "other_regulation": other[:10],
        }
        return ToolResult(output=out)

    # --- cross-reference data ---------------------------------------------------------------

    @cached_property
    def explicit_gdpr_links(self) -> dict[str, list[tuple[str, str]]]:
        """AI Act article -> [(GDPR article id, AI Act unit label that cites it)], from the text."""
        pattern = re.compile(
            r"Articles?\s+((?:\d+(?:\(\d+\))?(?:\s*(?:,|and|or)\s*)?)+)\s+of\s+Regulation\s+\(EU\)"
            r"\s+2016/679"
        )
        links: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for (lg, _), u in self.units.items():
            if (
                lg != "en"
                or u.regulation != "aiact"
                or u.type not in ("paragraph", "point", "definition")
            ):
                continue
            for m in pattern.finditer(u.text):
                for num in re.findall(r"(\d+)(?:\(\d+\))?", m.group(1)):
                    links[f"aiact:art:{u.article}"].append(
                        (f"gdpr:art:{num}", label_for(cite_id(u)))
                    )
        return links

    def _other_regulation(self, regulation: str, article: str, language: str) -> list[dict]:
        curated = json.loads((DATA / "cross_regulation.json").read_text("utf-8"))["pairs"]
        out = []
        if regulation == "aiact":
            for gdpr_id, where in self.explicit_gdpr_links.get(f"aiact:art:{article}", []):
                out.append(
                    {
                        **self._label_title(gdpr_id, language),
                        "basis": f"explicit reference in {where}",
                    }
                )
            for pair in curated:
                if pair["aiact"] == f"aiact:art:{article}":
                    out.append(
                        {
                            **self._label_title(pair["gdpr"], language),
                            "basis": f"curated: {pair['reason']}",
                        }
                    )
        else:
            target = f"gdpr:art:{article}"
            for ai_id, links in self.explicit_gdpr_links.items():
                for gdpr_id, where in links:
                    if gdpr_id == target:
                        out.append(
                            {
                                **self._label_title(ai_id, language),
                                "basis": f"explicit reference in {where}",
                            }
                        )
            for pair in curated:
                if pair["gdpr"] == target:
                    out.append(
                        {
                            **self._label_title(pair["aiact"], language),
                            "basis": f"curated: {pair['reason']}",
                        }
                    )
        return _dedupe(out)

    def _label_title(self, cid: str, language: str) -> dict:
        unit = self._unit_for_cite(cid, language)
        return {"label": label_for(cid), "title": unit.title if unit else None}

    def _unit_for_cite(self, cid: str, language: str) -> Unit | None:
        parts = cid.split(":")
        if parts[1] == "art" and len(parts) == 4 and DEFINITIONS_ARTICLE[parts[0]] == parts[2]:
            return self.units.get((language, f"{parts[0]}:def:{parts[3]}"))
        return self.units.get((language, cid))


def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for item in items:
        if item["label"] not in seen:
            seen.add(item["label"])
            out.append(item)
    return out


# --- validation and the runner -------------------------------------------------------------


def validation_message(name: str, exc: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors()
    )
    hints = {
        "language": "use the language of the question (en, fr or de)",
        "role": "role must be one of provider, deployer, importer, distributor, "
        "authorised_representative; if the user's role is unclear, ask",
        "category": "category must be high_risk, limited, gpai or gpai_systemic; prohibited "
        "systems have no obligations to list",
        "system_description": "give the system's purpose, the people it affects and where it is "
        "used (at least 30 characters)",
    }
    extra = "; ".join(h for key, h in hints.items() if key in problems)
    return f"Invalid arguments for {name}: {problems}." + (f" Fix: {extra}." if extra else "")


def run_tool(tools: Tools, name: str, raw_args: dict) -> ToolResult:
    """Validate the arguments and run one tool (no budget or allow-list: see ToolRunner)."""
    model = ARG_MODELS.get(name)
    if model is None:
        return _err("not_allowed", f"Unknown tool {name!r}. Tools: {', '.join(ARG_MODELS)}.")
    try:
        args = model.model_validate(raw_args)
    except ValidationError as exc:
        return _err("invalid_argument", validation_message(name, exc))
    return getattr(tools, name)(args)


def timed(fn, *args) -> tuple[ToolResult, float]:
    start = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - start) * 1000
