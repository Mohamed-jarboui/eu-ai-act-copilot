"""Cross-reference extraction in EN, FR and DE."""

import pytest

from copilot.ingest.refs import extract_cross_refs


@pytest.mark.parametrize(
    ("lang", "text", "expected"),
    [
        ("en", "as referred to in Article 9", ["art:9"]),
        ("en", "listed in Annex III and Article 9", ["art:9", "anx:III"]),
        ("en", "Article 9, Article 9 and Article 12", ["art:9", "art:12"]),  # no duplicates
        ("en", "Articles 9 and 10", ["art:9", "art:10"]),
        ("en", "Article 6(2) applies", ["art:6:2"]),
        ("fr", "visés à l’article 9", ["art:9"]),
        ("fr", "énumérés à l’annexe III", ["anx:III"]),
        ("fr", "aux articles 9 et 10", ["art:9", "art:10"]),
        ("de", "gemäß Artikel 9", ["art:9"]),
        ("de", "gemäß Anhang III", ["anx:III"]),
        ("de", "Artikel 6 Absatz 2", ["art:6:2"]),
        ("en", "Articles 8 to 15 apply", [f"art:{n}" for n in range(8, 16)]),  # ranges expand
        ("en", "Article 6 of Regulation (EU) 2016/679", []),  # another act: skipped
        ("en", "Article 10 of Directive (EU) 2016/680 and Article 5 of this Regulation", ["art:5"]),
        ("en", "based on Article 16 TFEU. In light of", []),
        ("en", "Annexes I and III", ["anx:I", "anx:III"]),
        ("en", "Article 43a applies", ["art:43a"]),
        ("fr", "l’article 6, paragraphe 2, et l’article 7", ["art:6:2", "art:7"]),
        ("fr", "l’article 6 du règlement (UE) 2016/679", []),
        ("fr", "l’article 5 du présent règlement", ["art:5"]),
        ("de", "Artikel 6 der Verordnung (EU) 2016/679", []),
        ("de", "die Artikel 9 und 10", ["art:9", "art:10"]),
        ("de", "Anhänge I und III", ["anx:I", "anx:III"]),
    ],
)
def test_extract_cross_refs(lang, text, expected):
    assert extract_cross_refs(text, lang) == expected
