"""Cross-reference extraction. The FR/DE and list cases are the TODO(me) targets: remove the
xfail marker from a case once your implementation handles it."""

import pytest

from copilot.ingest.refs import extract_cross_refs

TODO = pytest.mark.xfail(reason="TODO(me): extract_cross_refs", strict=False)


def todo(lang, text, expected):
    return pytest.param(lang, text, expected, marks=TODO)


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        ("en", "as referred to in Article 9", ["art:9"]),
        ("en", "listed in Annex III and Article 9", ["art:9", "anx:III"]),
        ("en", "Article 9, Article 9 and Article 12", ["art:9", "art:12"]),  # no duplicates
        todo("en", "Articles 9 and 10", ["art:9", "art:10"]),
        todo("en", "Article 6(2) applies", ["art:6:2"]),
        todo("fr", "visés à l’article 9", ["art:9"]),
        todo("fr", "énumérés à l’annexe III", ["anx:III"]),
        todo("fr", "aux articles 9 et 10", ["art:9", "art:10"]),
        todo("de", "gemäß Artikel 9", ["art:9"]),
        todo("de", "gemäß Anhang III", ["anx:III"]),
        todo("de", "Artikel 6 Absatz 2", ["art:6:2"]),
    ],
)
def test_extract_cross_refs(lang, text, expected):
    assert extract_cross_refs(text, lang) == expected
