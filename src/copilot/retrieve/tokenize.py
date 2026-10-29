"""Language-aware tokenization for BM25: fold case and accents, drop stopwords, strip suffixes.

The stemmers are deliberately crude suffix strippers (no NLTK/Snowball: dependencies stay light).
They only need to map the common inflections of a word to the same token, query and document alike.
"""

from __future__ import annotations

import re
import unicodedata

_STOPWORDS_RAW = {
    "en": """a an the of to in on at by for from with as and or not no is are was were be been
        it its this that these those which who whom what when where how why does do did can could
        shall may must should would will have has had any all such per under than then there
        their they them you your we our i me my""",
    "fr": """le la les l un une des du de d et ou en à au aux pour par sur dans que qu qui quoi
        est sont être a ont avoir ce cet cette ces se sa son ses leur leurs il elle on ne pas plus
        comment quand quel quelle quels quelles doit peut puis je j me m mon ma mes nous vous
        y c s t n il ils elles ainsi lors entre sous avec sans""",
    "de": """der die das den dem des ein eine einer eines einem einen und oder von zu zur zum im in
        an auf für mit nach bei aus als auch ist sind wird werden wurde wurden nicht sich es sie
        er wir ich mein meine meinen wenn was wer wie wo welche welcher welches welchen muss
        müssen darf kann können hat haben sein ihr ihre ihren über unter durch bis""",
}

_SUFFIXES = {
    "en": ("ations", "ation", "ings", "ing", "edly", "ed", "es", "s"),
    "fr": ("ements", "ement", "ations", "ation", "es", "s", "x", "e"),
    "de": ("ungen", "ung", "heiten", "heit", "keiten", "keit", "en", "er", "es", "e", "n", "s"),
}
_MIN_STEM = {"en": 3, "fr": 4, "de": 4}


def fold(text: str) -> str:
    """Lowercase and strip accents/umlauts (ä -> a, é -> e; ß -> ss via casefold)."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


STOPWORDS = {lang: {fold(w) for w in raw.split()} for lang, raw in _STOPWORDS_RAW.items()}


def stem(token: str, lang: str) -> str:
    for suffix in _SUFFIXES[lang]:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM[lang]:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, lang: str) -> list[str]:
    """Split on anything that is not a letter or digit (so "Hochrisiko-KI-Systeme" splits)."""
    tokens = re.findall(r"[a-z0-9]+", fold(text))
    return [stem(t, lang) for t in tokens if t not in STOPWORDS[lang]]
