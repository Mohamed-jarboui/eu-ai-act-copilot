"""The unit of the corpus: one citable piece of a regulation, in one language."""

from __future__ import annotations

from dataclasses import asdict, dataclass

UNIT_TYPES = (
    "recital",
    "article",
    "paragraph",
    "point",
    "definition",
    "annex",
    "annex_section",
)


@dataclass(frozen=True)
class Unit:
    """One corpus entry.

    `id` is language-independent (e.g. "aiact:art:6:2"), so the same id exists once per language;
    uniqueness is per (regulation, language). Hierarchy is given by `parent_id`.
    """

    id: str
    regulation: str  # "aiact" | "gdpr"
    language: str  # "en" | "fr" | "de"
    type: str  # one of UNIT_TYPES
    article: str | None  # article number, for article/paragraph/point/definition units
    annex: str | None  # annex numeral, for annex/annex_section units
    number: str  # this unit's own label: "6", "2", "a", "III", ...
    title: str | None
    chapter: str | None
    text: str
    refs: tuple[str, ...]  # normalised cross-reference targets, see refs.extract_cross_refs
    parent_id: str | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["refs"] = list(self.refs)
        return data
