"""Cross-reference extraction: find "Article 9" / "Annex III" style mentions in a unit's text."""

from __future__ import annotations

import re

# Placeholder pattern: English, whole-article and whole-annex mentions only.
_ARTICLE_EN = re.compile(r"\bArticles?\s+(\d+)")
_ANNEX_EN = re.compile(r"\bAnnex(?:es)?\s+([IVXLC]+)\b")


def extract_cross_refs(text: str, lang: str) -> list[str]:
    """Return the normalised targets mentioned in `text`, in order of first appearance, no dupes.

    Output format (language-independent, regulation-agnostic):
        "art:9"       an article           "art:6:2"   paragraph 2 of Article 6
        "anx:III"     an annex

    TODO(me): implement this for all three languages. The placeholder below only finds English
    "Article 9" and "Annex III", which is enough for the pipeline to run. Patterns to cover:

        EN  "Article 9"   "Articles 9 and 10"   "Article 6(2)"   "Annex III"   "Annexes I and III"
        FR  "article 9"   "articles 9 et 10"    "article 6, paragraphe 2"   "annexe III"
        DE  "Artikel 9"   "die Artikel 9 und 10"   "Artikel 6 Absatz 2"   "Anhang III"

    Hints:
      * Match case-insensitively and per language: the keyword differs (Article/article/Artikel,
        Annex/annexe/Anhang), and French/German use lowercase mid-sentence.
      * Lists ("Articles 9 and 10", "articles 9 et 10", "Artikel 9 und 10") hide a second number
        after "and"/"et"/"und", and sometimes commas: capture the whole list, then split it.
      * Decide what to do with ranges ("Articles 8 to 15") and with references to OTHER acts:
        "Article 6 of Regulation (EU) 2016/679" is not an article of the same text. A simple
        and honest choice is to skip a match followed by "of Regulation/Directive" and document it.
      * Add the cases you handle to tests/test_refs.py (the FR and DE ones are marked xfail now:
        remove the marker as they start passing).
    """
    refs: list[str] = []
    for pattern, prefix in ((_ARTICLE_EN, "art"), (_ANNEX_EN, "anx")):
        for match in pattern.finditer(text):
            ref = f"{prefix}:{match.group(1)}"
            if ref not in refs:
                refs.append(ref)
    return refs
