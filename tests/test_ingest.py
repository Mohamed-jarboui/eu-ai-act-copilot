"""Ingestion tests on the committed raw files (no network)."""

import hashlib
import json
import re
from pathlib import Path

import pytest

from copilot.ingest.corpus import build_corpus, check_unique, count_units, read_corpus
from copilot.ingest.download import LANGUAGES, REGULATIONS

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LANGS = list(LANGUAGES)

# Golden counts per (regulation, type), identical in EN/FR/DE. Taken from the `make ingest`
# output of 2026-10-29 (see docs/data-card.md); a parser change that moves them must be deliberate.
EXPECTED_COUNTS = {
    "aiact": {
        "recital": 180,
        "article": 113,
        "paragraph": 500,
        "point": 388,
        "definition": 68,
        "annex": 13,
        "annex_section": 91,
    },
    "gdpr": {
        "recital": 173,
        "article": 99,
        "paragraph": 372,
        "point": 352,
        "definition": 26,
        "annex": 0,
        "annex_section": 0,
    },
}


@pytest.fixture(scope="module")
def units():
    return build_corpus(RAW)


@pytest.fixture(scope="module")
def index(units):
    """(regulation, language) -> {id: Unit}"""
    out: dict = {}
    for u in units:
        out.setdefault((u.regulation, u.language), {})[u.id] = u
    return out


def children(index, reg, lang, parent_id, type_=None):
    return [
        u
        for u in index[(reg, lang)].values()
        if u.parent_id == parent_id and (type_ is None or u.type == type_)
    ]


def test_raw_files_match_manifest():
    manifest = json.loads((RAW / "manifest.json").read_text("utf-8"))
    assert len(manifest) == len(REGULATIONS) * len(LANGUAGES)
    for name, entry in manifest.items():
        body = (RAW / name).read_bytes()
        assert len(body) == entry["bytes"]
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("lang", LANGS)
def test_counts_are_stable(units, lang):
    counts = count_units(units)
    for reg, by_type in EXPECTED_COUNTS.items():
        for type_, expected in by_type.items():
            assert counts[(reg, lang, type_)] == expected, (reg, lang, type_)


def test_ids_unique_per_regulation_and_language(units):
    check_unique(units)  # raises on duplicates


@pytest.mark.parametrize("reg", list(REGULATIONS))
@pytest.mark.parametrize("lang", ["fr", "de"])
def test_ids_aligned_across_languages(index, reg, lang):
    assert set(index[(reg, lang)]) == set(index[(reg, "en")])


def test_every_parent_exists_and_no_text_is_empty(units, index):
    for u in units:
        assert u.text.strip(), u.id
        if u.parent_id:
            assert u.parent_id in index[(u.regulation, u.language)], u.id


def test_paragraph_text_has_no_leading_number(units):
    for u in units:
        if u.type == "paragraph":
            assert not re.match(r"^\(?\d+[.)]\s", u.text), (u.id, u.language)


def test_committed_corpus_is_up_to_date(units):
    assert read_corpus(ROOT / "data" / "processed" / "corpus.jsonl") == units


@pytest.mark.parametrize("lang", LANGS)
def test_aiact_article_5_has_its_points(index, lang):
    para1 = "aiact:art:5:1"
    points = children(index, "aiact", lang, para1, "point")
    assert [p.number for p in points] == list("abcdefgh")
    # point (c) has sub-points (i) and (ii)
    assert [p.number for p in children(index, "aiact", lang, f"{para1}:c", "point")] == ["i", "ii"]
    assert index[("aiact", lang)]["aiact:art:5"].article == "5"


@pytest.mark.parametrize("lang", LANGS)
def test_aiact_article_6_paragraphs(index, lang):
    paragraphs = children(index, "aiact", lang, "aiact:art:6", "paragraph")
    assert [p.number for p in paragraphs] == [str(n) for n in range(1, 9)]


@pytest.mark.parametrize(
    ("lang", "title"),
    [
        ("en", "Prohibited AI practices"),
        ("fr", "Pratiques interdites en matière d’IA"),
        ("de", "Verbotene Praktiken im KI-Bereich"),
    ],
)
def test_aiact_article_5_title_and_chapter(index, lang, title):
    article = index[("aiact", lang)]["aiact:art:5"]
    assert article.title == title
    assert article.chapter.split()[1] == "II"  # CHAPTER II / CHAPITRE II / KAPITEL II


@pytest.mark.parametrize(
    ("reg", "lang", "term"),
    [
        ("aiact", "en", "AI system"),
        ("aiact", "fr", "système d’IA"),
        ("aiact", "de", "KI-System"),
        ("gdpr", "en", "personal data"),
        ("gdpr", "fr", "données à caractère personnel"),
        ("gdpr", "de", "personenbezogene Daten"),
    ],
)
def test_first_definition(index, reg, lang, term):
    definition = index[(reg, lang)][f"{reg}:def:1"]
    assert definition.type == "definition"
    assert definition.title == term


@pytest.mark.parametrize("lang", LANGS)
def test_annex_iii_sections(index, lang):
    sections = children(index, "aiact", lang, "aiact:anx:III", "annex_section")
    assert [s.number for s in sections] == [str(n) for n in range(1, 9)]


@pytest.mark.parametrize("lang", LANGS)
def test_recitals_are_numbered_without_label(index, lang):
    recital = index[("aiact", lang)]["aiact:rct:1"]
    assert recital.type == "recital"
    assert not recital.text.startswith("(1)")
