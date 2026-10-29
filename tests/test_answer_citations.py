"""Citation parsing, labels, sentence splitting, refusal and language detection.

Covers src/copilot/answer/citations.py, labels.py, refusal.py and language.py.
"""

import pytest

from copilot.answer import citations, refusal
from copilot.answer.evidence import Evidence
from copilot.answer.impl import get_impl
from copilot.answer.labels import cite_id, label_for
from copilot.answer.language import detect_language
from copilot.retrieve.chunks import Corpus


@pytest.fixture
def impl():
    return get_impl()


# --- parse_citation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("AI Act Art. 6(2)", "aiact:art:6:2"),
        ("GDPR Art. 22(1)", "gdpr:art:22:1"),
        ("AI Act Art. 5(1)(f)", "aiact:art:5:1:f"),
        ("AI Act Art. 5(1)(c)(ii)", "aiact:art:5:1:c:ii"),
        ("AI Act Art. 6", "aiact:art:6"),
        ("AI Act Art. 43a", "aiact:art:43a"),
        ("GDPR Art. 4(11)", "gdpr:art:4:11"),
        ("AI Act Annex III", "aiact:anx:III"),
        ("AI Act Annex III, point 1", "aiact:anx:III:1"),
        ("ai act art 6 (2)", "aiact:art:6:2"),  # lenient about case and spacing
        ("Directive 2016/680 Art. 3", None),  # not one of our regulations
        ("see above", None),
    ],
)
def test_parse_citation(impl, label, expected):
    assert impl.parse_citation(label) == expected


def test_parse_citation_stretch_annex_sections(impl):
    assert impl.parse_citation("AI Act Annex VIII, Section A, point 1") == "aiact:anx:VIII:A:1"


def test_every_corpus_label_round_trips(impl):
    """Parse the label of every indexed chunk back to its cite id: finds forgotten patterns."""
    chunks = Corpus.load().by_language["en"]
    failures = [
        (cite_id(u), label_for(cite_id(u)))
        for u in chunks
        if impl.parse_citation(label_for(cite_id(u))) != cite_id(u)
    ]
    assert failures == []


def test_cite_ids_and_labels_are_unique_in_the_corpus():
    ids = [cite_id(u) for u in Corpus.load().by_language["en"]]
    assert len(ids) == len(set(ids))
    assert len({label_for(i) for i in ids}) == len(ids)


# --- labels ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cid", "label"),
    [
        ("aiact:art:6:2", "AI Act Art. 6(2)"),
        ("gdpr:art:22:1", "GDPR Art. 22(1)"),
        ("aiact:art:16:a", "AI Act Art. 16(a)"),
        ("aiact:anx:III", "AI Act Annex III"),
        ("aiact:anx:III:1", "AI Act Annex III, point 1"),
        ("aiact:anx:I:A", "AI Act Annex I, Section A"),
        ("aiact:anx:XI:1:2", "AI Act Annex XI, Section 1, point 2"),
    ],
)
def test_label_for(cid, label):
    assert label_for(cid) == label


def test_a_definition_is_cited_as_its_article(make_unit):
    unit = make_unit("aiact:def:3", "‘provider’ means ...", type="definition", article="3")
    assert cite_id(unit) == "aiact:art:3:3"
    gdpr = make_unit("gdpr:def:1", "‘personal data’", regulation="gdpr", type="definition")
    assert label_for(cite_id(gdpr)) == "GDPR Art. 4(1)"


def test_second_lists_lose_their_list_marker(make_unit):
    assert cite_id(make_unit("aiact:art:43:1:l2:a", "x")) == "aiact:art:43:1:a"


# --- finding citations and matching them to sources --------------------------------------------


def test_find_citations_in_order_and_with_several_per_bracket():
    found = citations.find_citations(
        "A [AI Act Art. 5(1)]. B [GDPR Art. 22(1); AI Act Art. 86] and [1] and [sic].",
        parse=get_impl().parse_citation,
    )
    assert [(f.raw, f.cite_id) for f in found] == [
        ("AI Act Art. 5(1)", "aiact:art:5:1"),
        ("GDPR Art. 22(1)", "gdpr:art:22:1"),
        ("AI Act Art. 86", "aiact:art:86"),
    ]


def test_unparsable_citation_is_kept_with_no_id():
    found = citations.find_citations("X [GDPR Art. nonsense].", parse=lambda _label: None)
    assert [(f.raw, f.cite_id) for f in found] == [("GDPR Art. nonsense", None)]


@pytest.mark.parametrize(
    ("cited", "source", "expected"),
    [
        ("aiact:art:6:2", "aiact:art:6:2", True),
        ("aiact:art:6", "aiact:art:6:2", True),  # citing the article is backed by its paragraph
        ("aiact:art:6:2:a", "aiact:art:6:2", True),  # citing a point is backed by its paragraph
        ("aiact:art:6:1", "aiact:art:6:2", False),  # siblings never support each other
        ("aiact:art:60", "aiact:art:6", False),  # prefix of the number is not containment
        ("gdpr:art:6:2", "aiact:art:6:2", False),
    ],
)
def test_supports(cited, source, expected):
    assert citations.supports(cited, source) is expected


def test_split_sentences_keeps_citations_attached():
    text = "Providers must log events [AI Act Art. 12(1)]. They keep logs. [AI Act Art. 19(1)]"
    assert citations.split_sentences(text) == [
        "Providers must log events [AI Act Art. 12(1)].",
        "They keep logs. [AI Act Art. 19(1)]",
    ]


def test_split_sentences_does_not_split_inside_a_citation():
    assert len(citations.split_sentences("See [AI Act Art. 6(2)]. Done.")) == 2


# --- refusal ---------------------------------------------------------------------------------


def evidence(**kw):
    base = dict(
        mode="hybrid", n_sources=5, top_cosine=0.7, top_bm25=9.0, query_tokens=6, language="en"
    )
    return Evidence(**{**base, **kw})


def test_refuse_when_nothing_was_retrieved(impl):
    assert impl.should_refuse(evidence(n_sources=0, top_cosine=None, top_bm25=None))


def test_do_not_refuse_with_sources_and_a_good_score(impl):
    assert not impl.should_refuse(evidence())


def test_english_cosine_gate(impl):
    threshold = refusal.COSINE_MIN["en"]
    assert impl.should_refuse(evidence(top_cosine=threshold - 0.01))
    assert not impl.should_refuse(evidence(top_cosine=threshold + 0.01))


def test_bm25_mode_is_never_refused_on_score(impl):
    """No cosine exists, and BM25 scores cannot tell out-of-scope from in-scope questions."""
    assert not impl.should_refuse(evidence(mode="bm25", top_cosine=None, top_bm25=0.1))


def test_languages_without_a_threshold_are_not_gated(impl):
    assert not impl.should_refuse(evidence(language="fr", top_cosine=0.01))


def test_the_threshold_separates_the_calibration_data():
    """The measured values behind COSINE_MIN: 9 in-scope English gold questions and the GPU one."""
    in_scope = [0.781, 0.749, 0.699, 0.783, 0.745, 0.740, 0.814, 0.773, 0.719]
    out_of_scope = [0.599]
    assert all(c >= refusal.COSINE_MIN["en"] for c in in_scope)
    assert all(c < refusal.COSINE_MIN["en"] for c in out_of_scope)


# --- language ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "lang"),
    [
        ("Does a chatbot have to tell people they are talking to a machine?", "en"),
        ("Quelles obligations a une entreprise qui utilise un système d'IA à haut risque ?", "fr"),
        ("Müssen Deepfakes gekennzeichnet werden?", "de"),
        ("Welche Pflichten haben Anbieter von Hochrisiko-KI-Systemen?", "de"),
        ("À partir de quand le règlement sur l'IA s'applique-t-il ?", "fr"),
        ("GDPR", "en"),  # nothing to go on: English
    ],
)
def test_detect_language(question, lang):
    assert detect_language(question) == lang
