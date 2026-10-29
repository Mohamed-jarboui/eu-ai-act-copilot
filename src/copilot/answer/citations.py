"""Finding citations in an answer and tying them back to the sources the model was given."""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_BRACKET = re.compile(r"\[([^\[\]]+)\]")


_REGULATION = {"ai act": "aiact", "gdpr": "gdpr"}
_ARTICLE = re.compile(
    r"^(AI\s*Act|GDPR)\s+Art(?:icle)?\.?\s*(\d+[a-z]?)\s*((?:\(\s*\w+\s*\)\s*)*)$", re.IGNORECASE
)
_ANNEX = re.compile(
    r"^(AI\s*Act|GDPR)\s+Annex\s+([IVXLC]+)"
    r"(?:\s*,\s*Section\s+(\w+))?(?:\s*,\s*point\s+\(?(\w+)\)?)?$",
    re.IGNORECASE,
)


def parse_citation(label: str) -> str | None:
    """Map ONE citation label, the text inside the brackets, to a cite id; None if not recognised.

    Examples (the inverse of `labels.label_for`):

        "AI Act Art. 6(2)"                      -> "aiact:art:6:2"
        "GDPR Art. 22(1)"                       -> "gdpr:art:22:1"
        "AI Act Art. 5(1)(f)"                   -> "aiact:art:5:1:f"
        "AI Act Art. 6"                         -> "aiact:art:6"
        "AI Act Annex III"                      -> "aiact:anx:III"
        "AI Act Annex III, point 1"             -> "aiact:anx:III:1"
        "AI Act Annex VIII, Section A, point 1" -> "aiact:anx:VIII:A:1"
        "Directive 2016/680 Art. 3"             -> None   (not one of our two regulations)

    Lenient about case and spacing ("ai act art 6 (2)"), because models are not perfectly
    consistent. Anything not recognised returns None, which the pipeline treats as an invalid
    citation and drops the sentence, rather than guessing an id.
    """
    label = " ".join(label.split())
    if match := _ARTICLE.match(label):
        regulation = _REGULATION[re.sub(r"\s+", " ", match.group(1).lower())]
        parts = re.findall(r"\(\s*(\w+)\s*\)", match.group(3))
        return ":".join([regulation, "art", match.group(2).lower(), *[p.lower() for p in parts]])
    if match := _ANNEX.match(label):
        regulation = _REGULATION[re.sub(r"\s+", " ", match.group(1).lower())]
        parts = [match.group(2).upper()]
        if match.group(3):
            parts.append(match.group(3).upper())
        if match.group(4):
            parts.append(match.group(4).lower())
        return ":".join([regulation, "anx", *parts])
    return None


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
