"""Citation ids and labels: how a corpus unit is named in an answer.

A *cite id* is the id a lawyer would cite, built from the unit id:
    aiact:art:6:2        -> label "AI Act Art. 6(2)"
    gdpr:art:22:1        -> label "GDPR Art. 22(1)"
    aiact:art:5:1:f      -> label "AI Act Art. 5(1)(f)"
    aiact:anx:III        -> label "AI Act Annex III"
    aiact:anx:III:1      -> label "AI Act Annex III, point 1"
    aiact:anx:VIII:A:1   -> label "AI Act Annex VIII, Section A, point 1"
A definition (`aiact:def:3`) is cited as the article that holds it (`aiact:art:3:3`, "Art. 3(3)").
Second lists that reuse labels (`...:43:1:l2:a`) lose their "l2" in the label, so the label of
those points is ambiguous: the answer cites the first list.
"""

from __future__ import annotations

import re

from copilot.ingest.models import Unit
from copilot.ingest.parse import DEFINITIONS_ARTICLE

REGULATIONS = {"aiact": "AI Act", "gdpr": "GDPR"}


def cite_id(unit: Unit) -> str:
    """The id under which a unit is cited in an answer."""
    if unit.type == "definition":
        return f"{unit.regulation}:art:{DEFINITIONS_ARTICLE[unit.regulation]}:{unit.number}"
    return re.sub(r":l\d+(?=:)", "", unit.id)


def label_for(cid: str) -> str:
    """The label shown to the model and the reader for a cite id (see the module docstring)."""
    reg, kind, *rest = cid.split(":")
    name = REGULATIONS[reg]
    if kind == "art":
        number, *parts = rest
        return f"{name} Art. {number}" + "".join(f"({p})" for p in parts)
    numeral, *parts = rest
    label = f"{name} Annex {numeral}"
    if len(parts) == 2:  # section, then the point inside it
        return label + f", Section {parts[0]}, point {parts[1]}"
    if len(parts) == 1:  # a lone part is a section if it is a capital letter ("A"), else a point
        part = parts[0]
        return label + (
            f", Section {part}" if part.isalpha() and part.isupper() else f", point {part}"
        )
    return label
