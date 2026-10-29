"""Cross-reference extraction: find "Article 9" / "Annex III" style mentions in a unit's text.

Output format (language-independent, regulation-agnostic), in order of first appearance, no dupes:

    "art:9"      an article          "art:6:2"   paragraph 2 of Article 6
    "anx:III"    an annex

Patterns covered:

    EN  "Article 9"  "Articles 9 and 10"  "Article 6(2)"  "Annex III"  "Annexes I and III"
    FR  "article 9"  "articles 9 et 10"   "article 6, paragraphe 2"   "annexe III"
    DE  "Artikel 9"  "die Artikel 9 und 10"   "Artikel 6 Absatz 2"   "Anhang III"

Decisions (see docs/data-card.md):
  * Keywords are case-insensitive and per language; Roman numerals stay case-sensitive.
  * A list ("Articles 9 and 10") is captured whole and split. A range ("Articles 8 to 15") is
    expanded to every article in it.
  * A reference to ANOTHER act ("Article 6 of Regulation (EU) 2016/679", "Article 16 TFEU") is
    skipped, because "art:6" would point at Article 6 of this text instead. "of this Regulation"
    is kept. The target regulation of a skipped reference is not recorded.
  * Not captured: sub-paragraph and point levels ("first subparagraph, point (g)").
"""

from __future__ import annotations

import re

# Per language: article keyword, annex keyword, "paragraph" word, "and/or" words, "to" (range).
_KW = {
    "en": dict(art=r"Articles?", anx=r"Annex(?:es)?", para=r"paragraph", conj=r"and|or", to=r"to"),
    "fr": dict(art=r"articles?", anx=r"annexes?", para=r"paragraphe", conj=r"et|ou", to=r"à"),
    "de": dict(
        art=r"Artikel[n]?", anx=r"Anh(?:ang|änge[n]?)", para=r"Absatz", conj=r"und|oder", to=r"bis"
    ),
}

# "... of Regulation (EU) 2016/679" and similar: the target is NOT in this text.
_OTHER_ACT = {
    "en": (
        r"of\s+(?:the\s+)?(?:Council\s+|Commission\s+)?"
        r"(?:(?:Delegated\s+|Implementing\s+)?(?:Regulation|Directive|Decision)"
        r"|Treaty|TFEU|that\s+(?:Regulation|Directive))\b"
    ),
    "fr": r"(?:du|de\s+la|des|de\s+l’)\s*(?:règlement|directive|décision|traité|TFUE)\b",
    "de": r"(?:der|des)\s+(?:Verordnung|Richtlinie|Beschlusses|Entscheidung|Vertrags?|AEUV)\b",
}
_TREATY_ABBREV = r"(?:TFEU|TEU|TFUE|TUE|AEUV|EUV)\b"
# The clause that follows a reference ends at a sentence break or at the next Article/Annex.
_CLAUSE_END = r"[.;]\s|\b(?i:articles?|artikel\w*|annex\w*|anh\w+)\b"


def _build(lang: str, kind: str) -> re.Pattern:
    k = _KW[lang]
    sep = rf"(?:\s*,\s*(?:(?i:{k['conj']})\s+)?|\s+(?i:{k['conj']})\s+|\s+(?i:{k['to']})\s+)"
    if kind == "art":
        item = rf"\d+[a-z]?(?:\(\d+\)|,?\s*(?i:{k['para']})\s+\d+)?"
    else:
        item = r"[IVXLC]+\b"
    return re.compile(rf"(?i:{k[kind]})\s+(?P<span>{item}(?:{sep}{item})*)")


_PATTERNS = {(lang, kind): _build(lang, kind) for lang in _KW for kind in ("art", "anx")}
_TOKEN = {
    lang: re.compile(rf"(?P<n>\d+[a-z]?)(?:\((?P<p1>\d+)\)|,?\s*(?i:{k['para']})\s+(?P<p2>\d+))?")
    for lang, k in _KW.items()
}


def _is_other_act(tail: str, lang: str) -> bool:
    """Does the text right after a reference say it belongs to another act (Directive, TFEU...)?"""
    clause = re.split(_CLAUSE_END, tail, maxsplit=1)[0][:90]
    if re.match(rf"\s*{_TREATY_ABBREV}", clause):
        return True
    return bool(re.search(_OTHER_ACT[lang], clause, re.IGNORECASE))


def _article_refs(span: str, lang: str) -> list[str]:
    refs, prev = [], None
    tokens = list(_TOKEN[lang].finditer(span))
    for i, tok in enumerate(tokens):
        num, para = tok.group("n"), tok.group("p1") or tok.group("p2")
        ref = f"art:{num}:{para}" if para else f"art:{num}"
        between = span[tokens[i - 1].end() : tok.start()] if i else ""
        is_range = re.fullmatch(rf"\s*(?i:{_KW[lang]['to']})\s+", between)
        if i and is_range and prev and num.isdigit():  # "Articles 8 to 15" -> 8, 9, ..., 15
            refs.extend(f"art:{n}" for n in range(int(prev) + 1, int(num)))
        refs.append(ref)
        prev = num if num.isdigit() and not para else None
    return refs


def extract_cross_refs(text: str, lang: str) -> list[str]:
    """Return the normalised targets mentioned in `text` (see the module docstring)."""
    refs: list[str] = []
    for kind in ("art", "anx"):
        for m in _PATTERNS[(lang, kind)].finditer(text):
            if _is_other_act(text[m.end() :], lang):
                continue  # refers to another act
            found = (
                _article_refs(m.group("span"), lang)
                if kind == "art"
                else [f"anx:{r}" for r in re.findall(r"[IVXLC]+\b", m.group("span"))]
            )
            refs.extend(r for r in found if r not in refs)
    return refs
