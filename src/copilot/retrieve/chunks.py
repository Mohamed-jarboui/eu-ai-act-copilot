"""Which corpus units are indexed for retrieval, and how a unit becomes searchable text.

The corpus overlaps on purpose (article > paragraph > point). Retrieval needs ONE granularity with
no overlap, so a chunk is the finest unit that still reads on its own:

  * paragraph          for articles with numbered paragraphs (points stay inline in the paragraph)
  * article            for the unnumbered ones (no paragraph units)
  * definition         one per definition; the definitions article itself is not indexed as a block
  * annex_section      for annexes that were split into sections
  * annex              only for annexes without sections
  * recital            only if asked for (they never count as hits in the eval)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from copilot.ingest.corpus import read_corpus
from copilot.ingest.models import Unit
from copilot.ingest.parse import DEFINITIONS_ARTICLE

DEFAULT_TYPES = frozenset({"paragraph", "definition", "annex_section", "article", "annex"})
ALL_TYPES = DEFAULT_TYPES | {"recital"}


def select_chunks(units: Iterable[Unit], types: frozenset[str] = DEFAULT_TYPES) -> list[Unit]:
    units = list(units)
    has_paragraphs = {(u.regulation, u.language, u.article) for u in units if u.type == "paragraph"}
    has_sections = {(u.regulation, u.language, u.annex) for u in units if u.type == "annex_section"}
    chunks = []
    for u in units:
        if u.type not in types:
            continue
        if u.type == "article" and (
            (u.regulation, u.language, u.article) in has_paragraphs
            or DEFINITIONS_ARTICLE.get(u.regulation) == u.article
        ):
            continue
        if u.type == "annex" and (u.regulation, u.language, u.annex) in has_sections:
            continue
        chunks.append(u)
    return chunks


def chunk_text(unit: Unit) -> str:
    """Text used for both BM25 and embeddings: the heading gives a bare paragraph its context."""
    if unit.title and unit.type != "definition":
        return f"{unit.title}\n{unit.text}"
    return unit.text


def anchor_id(unit: Unit) -> str:
    """The id a retrieved chunk is matched against the gold ids with.

    A definition (`aiact:def:1`) is part of the definitions article (`aiact:art:3`), so it is
    matched as that article. Every other chunk is matched by its own id (the README rule).
    """
    return unit.parent_id if unit.type == "definition" and unit.parent_id else unit.id


@dataclass(frozen=True)
class Filters:
    """Metadata filters. `language` is required: each language has its own tokenizer and index."""

    language: str
    regulation: str | None = None
    types: frozenset[str] | None = None

    def accepts(self, unit: Unit) -> bool:
        return (
            unit.language == self.language
            and (self.regulation is None or unit.regulation == self.regulation)
            and (self.types is None or unit.type in self.types)
        )


class Corpus:
    """Chunks grouped by language, in a stable order shared by the BM25 and vector indexes."""

    def __init__(self, units: Iterable[Unit], types: frozenset[str] = DEFAULT_TYPES) -> None:
        self.types = types
        self.by_language: dict[str, list[Unit]] = {}
        for unit in select_chunks(units, types):
            self.by_language.setdefault(unit.language, []).append(unit)

    @classmethod
    def load(cls, types: frozenset[str] = DEFAULT_TYPES) -> Corpus:
        return cls(read_corpus(), types)

    def mask(self, filters: Filters) -> list[bool]:
        return [filters.accepts(u) for u in self.by_language.get(filters.language, [])]
