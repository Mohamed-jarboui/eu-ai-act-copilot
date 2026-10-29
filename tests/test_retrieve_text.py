"""Chunking, tokenization and BM25."""

import pytest

from copilot.ingest.corpus import read_corpus
from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.chunks import (
    ALL_TYPES,
    Corpus,
    Filters,
    anchor_id,
    chunk_text,
    select_chunks,
)
from copilot.retrieve.tokenize import fold, tokenize

# --- tokenizer ---------------------------------------------------------------------------------


def test_fold_removes_accents_and_case():
    assert fold("Données ÉTÉ Ärger Straße") == "donnees ete arger strasse"


def test_english_drops_stopwords_and_plural_s():
    assert tokenize("What are the obligations of providers?", "en") == ["oblig", "provider"]


def test_german_splits_hyphenated_compounds():
    assert tokenize("Hochrisiko-KI-Systeme", "de") == ["hochrisiko", "ki", "system"]


def test_french_folds_accents_and_stems_plurals():
    assert tokenize("Les données personnelles", "fr") == tokenize("la donnee personnelle", "fr")


def test_singular_plural_and_apostrophe_styles_meet():
    assert tokenize("système d’IA", "fr") == tokenize("systèmes d'IA", "fr")


# --- chunks on the real corpus -----------------------------------------------------------------


@pytest.fixture(scope="module")
def units():
    return read_corpus()


def test_chunk_counts_per_language(units):
    chunks = select_chunks(units)
    for lang in ("en", "fr", "de"):
        assert sum(1 for u in chunks if u.language == lang) == 1092


def test_chunks_do_not_overlap(units):
    chunks = select_chunks(units)
    chunk_keys = {(u.regulation, u.language, u.id) for u in chunks}
    by_key = {(u.regulation, u.language, u.id): u for u in units}
    for u in chunks:
        parent = u.parent_id  # no ancestor of a chunk may itself be a chunk
        while parent:
            assert (u.regulation, u.language, parent) not in chunk_keys, (u.id, parent)
            parent = by_key[(u.regulation, u.language, parent)].parent_id


def test_definitions_article_is_not_indexed_as_a_block(units):
    chunk_ids = {u.id for u in select_chunks(units)}
    assert "aiact:art:3" not in chunk_ids and "gdpr:art:4" not in chunk_ids
    assert "aiact:def:1" in chunk_ids and "aiact:art:5:1" in chunk_ids


def test_recitals_only_when_asked(units):
    assert not any(u.type == "recital" for u in select_chunks(units))
    assert any(u.type == "recital" for u in select_chunks(units, ALL_TYPES))


def test_a_definition_is_matched_as_its_article(units):
    definition = next(u for u in units if u.id == "aiact:def:1" and u.language == "en")
    assert anchor_id(definition) == "aiact:art:3"
    paragraph = next(u for u in units if u.id == "aiact:art:5:1" and u.language == "en")
    assert anchor_id(paragraph) == "aiact:art:5:1"


def test_chunk_text_prepends_the_heading(make_unit):
    para = make_unit(
        "aiact:art:5:1", "The following are prohibited.", title="Prohibited AI practices"
    )
    assert chunk_text(para).startswith("Prohibited AI practices\n")
    definition = make_unit(
        "aiact:def:1", "‘AI system’ means ...", type="definition", title="AI system"
    )
    assert chunk_text(definition) == "‘AI system’ means ..."


# --- BM25 --------------------------------------------------------------------------------------


@pytest.fixture
def small_corpus(make_unit):
    return Corpus(
        [
            make_unit("aiact:art:5:1", "prohibited practices subliminal manipulation", article="5"),
            make_unit(
                "aiact:art:50:1",
                "chatbot users must be informed they interact with a machine",
                article="50",
            ),
            make_unit(
                "gdpr:art:33:1",
                "breach notification within seventy-two hours",
                regulation="gdpr",
                article="33",
            ),
            make_unit(
                "aiact:art:5:1", "verbotene praktiken manipulation", language="de", article="5"
            ),
            # BM25 gives a term in a one-document corpus a negative weight, so add other documents
            make_unit(
                "aiact:art:50:1", "chatbot muss sich als maschine zu erkennen geben", language="de"
            ),
            make_unit(
                "gdpr:art:33:1",
                "meldung einer verletzung binnen zweiundsiebzig stunden",
                language="de",
                regulation="gdpr",
            ),
        ]
    )


def test_bm25_ranks_the_matching_chunk_first(small_corpus):
    index = BM25Index(small_corpus)
    assert (
        index.search("which practices are prohibited", Filters(language="en"))[0] == "aiact:art:5:1"
    )
    assert index.search("chatbot informed", Filters(language="en"))[0] == "aiact:art:50:1"


def test_bm25_drops_chunks_with_no_matching_token(small_corpus):
    assert BM25Index(small_corpus).search("chatbot", Filters(language="en")) == ["aiact:art:50:1"]


def test_bm25_applies_the_regulation_filter(small_corpus):
    index = BM25Index(small_corpus)
    assert index.search("notification manipulation", Filters(language="en", regulation="gdpr")) == [
        "gdpr:art:33:1"
    ]


def test_bm25_is_per_language(small_corpus):
    assert BM25Index(small_corpus).search("manipulation", Filters(language="de")) == [
        "aiact:art:5:1"
    ]
