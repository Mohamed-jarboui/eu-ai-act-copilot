"""Finding citations in an answer and tying them back to the sources the model was given."""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_BRACKET = re.compile(r"\[([^\[\]]+)\]")


def parse_citation(label: str) -> str | None:
    """Map ONE citation label, the text inside the brackets, to a cite id; None if not recognised.

    Examples (the inverse of `labels.label_for`):

        "AI Act Art. 6(2)"                      -> "aiact:art:6:2"
        "GDPR Art. 22(1)"                       -> "gdpr:art:22:1"
        "AI Act Art. 5(1)(f)"                   -> "aiact:art:5:1:f"
        "AI Act Art. 6"                         -> "aiact:art:6"
        "AI Act Annex III"                      -> "aiact:anx:III"
        "AI Act Annex III, point 1"             -> "aiact:anx:III:1"
        "AI Act Annex VIII, Section A, point 1" -> "aiact:anx:VIII:A:1"   (stretch goal)
        "Directive 2016/680 Art. 3"             -> None   (not one of our two regulations)

    TODO(me): implement this.
    Hints:
      * Two regulation names: "AI Act" -> "aiact", "GDPR" -> "gdpr". Be lenient about case and
        spacing ("Art 6(2)", "Art.6 (2)"), because models are not perfectly consistent.
      * After the article number come zero or more parenthesised parts "(2)", "(f)", "(i)": each
        becomes one ":part", lowercased. A regex with a repeated group, or re.findall over the
        tail, does it. Article numbers can carry a letter ("43a").
      * Return None for anything you do not recognise: the pipeline treats that as an invalid
        citation (and drops the sentence), which is better than guessing an id.
      * Check yourself with tests/test_answer_citations.py: it also round-trips EVERY id of the
        real corpus through label_for and back, so any pattern you forgot shows up there.
    """
    raise NotImplementedError("TODO(me): implement the citation parser (see the docstring)")


@dataclass(frozen=True)
class FoundCitation:
    raw: str  # the text between the brackets, e.g. "AI Act Art. 6(2)"
    cite_id: str | None  # None if the label could not be parsed


def find_citations(text: str, parse=parse_citation) -> list[FoundCitation]:
    """All bracketed citations in `text`, in order. A bracket may hold several, split by ';'.

    Only brackets that start with a regulation name are considered, so other bracketed text such
    as "[1]" or "[sic]" is ignored.
    """
    found = []
    for match in CITATION_BRACKET.finditer(text):
        for part in match.group(1).split(";"):
            part = part.strip()
            if re.match(r"(?i)(AI Act|GDPR)\b", part):
                found.append(FoundCitation(part, parse(part)))
    return found


def supports(cited: str, source: str) -> bool:
    """Does a source (a cite id) back a cited id? Equal, or one contains the other.

    Citing "Art. 6" is backed by a source "Art. 6(2)" and citing "Art. 6(2)(a)" by a source
    "Art. 6(2)"; siblings ("Art. 6(1)" vs "Art. 6(2)") never support each other.
    """
    return cited == source or source.startswith(cited + ":") or cited.startswith(source + ":")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences, keeping bracketed citations attached to their sentence."""
    hidden: dict[str, str] = {}

    def hide(match: re.Match) -> str:
        key = f"\x00{len(hidden)}\x00"
        hidden[key] = match.group(0)
        return key

    protected = CITATION_BRACKET.sub(hide, text)
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", protected) if p.strip()]
    sentences: list[str] = []
    for piece in pieces:
        only_citations = not re.sub(r"\x00\d+\x00|[\s.,;]", "", piece)
        if only_citations and sentences:  # "... prohibited. [AI Act Art. 5]" -> attach back
            sentences[-1] += " " + piece
        else:
            sentences.append(piece)
    restored = []
    for sentence in sentences:
        for key, value in hidden.items():
            sentence = sentence.replace(key, value)
        restored.append(sentence)
    return restored
