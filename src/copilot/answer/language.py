"""Guess the language of a question (English, French or German) from its stopwords."""

from __future__ import annotations

import re

from copilot.retrieve.tokenize import STOPWORDS, fold

# Words that belong to one language only ("in", "a", "es" are shared and would only add noise).
_UNIQUE = {
    lang: words - set().union(*(w for other, w in STOPWORDS.items() if other != lang))
    for lang, words in STOPWORDS.items()
}
_ACCENT_HINTS = {"de": "äöüß", "fr": "àâçéèêëîïôùûœ"}


def detect_language(text: str) -> str:
    """Return "en", "fr" or "de"; English when nothing points elsewhere."""
    lowered = text.casefold()
    tokens = re.findall(r"[^\W\d_]+", fold(text))
    scores = {lang: sum(1 for t in tokens if t in words) for lang, words in _UNIQUE.items()}
    for lang, letters in _ACCENT_HINTS.items():
        scores[lang] += sum(1 for ch in lowered if ch in letters) * 0.5
    best = max(scores, key=lambda lang: (scores[lang], lang == "en"))
    return best if scores[best] > 0 else "en"
