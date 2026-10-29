"""The answer pipeline with fake models: happy path, refusals, grounding drops, streaming."""

from answer_support import ANSWER, DE_Q, FR_Q, HIGH_RISK_Q, build_pipeline
from copilot.answer.grounding import parse_verdicts


def test_happy_path_returns_checked_answer_with_citations(corpus):
    pipeline, gen, check = build_pipeline(corpus, [ANSWER], ["[1, 1]"])
    answer = pipeline.ask(HIGH_RISK_Q)

    assert not answer.refused and answer.grounded and answer.grounding_check == "passed"
    assert answer.answer == ANSWER
    assert [c.label for c in answer.citations] == ["AI Act Art. 6(2)", "AI Act Art. 5(1)"]
    assert answer.citations[0].id == "aiact:art:6:2"
    assert "Annex III" in answer.citations[0].snippet
    assert answer.language == "en" and answer.retrieval_mode == "bm25"
    assert answer.retrieval_notes  # says why there was no vector search
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == (200, 20)
    assert answer.usage.model_calls == 2 and answer.usage.total_ms > 0


def test_the_prompt_lists_numbered_sources_and_the_language(corpus):
    pipeline, gen, _ = build_pipeline(corpus, [ANSWER], ["[1, 1]"])
    pipeline.ask(HIGH_RISK_Q)
    prompt = gen.prompts[0]
    assert "[1] AI Act Art. 6(2) (Classification rules" in prompt
    assert "answer in English" in prompt and "NOT_COVERED" in prompt


def test_the_language_can_be_forced(corpus):
    pipeline, gen, _ = build_pipeline(corpus, ["NOT_COVERED"])
    answer = pipeline.ask(HIGH_RISK_Q, "fr")  # an English question, answered in French
    assert answer.language == "fr" and "answer in French" in gen.prompts[0]


def test_the_language_is_detected_and_retrieval_uses_that_corpus(corpus):
    pipeline, gen, _ = build_pipeline(corpus, ["NOT_COVERED"])
    answer = pipeline.ask(FR_Q)
    assert answer.language == "fr" and "answer in French" in gen.prompts[0]
    assert "Règles de classification" in gen.prompts[0]  # French chunks, not English ones


def test_not_covered_marker_becomes_a_refusal_in_the_users_language(corpus):
    pipeline, _, check = build_pipeline(corpus, ["NOT_COVERED"])
    answer = pipeline.ask(DE_Q)
    assert answer.refused and answer.refusal_reason == "not_covered"
    assert answer.answer == "Die Texte behandeln diese Frage nicht."
    assert answer.citations == [] and check.prompts == []  # no grounding call needed


def test_no_sources_refuses_without_calling_any_model(corpus):
    pipeline, gen, check = build_pipeline(corpus, [])
    answer = pipeline.ask("zzz qqq xxx")
    assert answer.refused and answer.refusal_reason == "no_sources"
    assert gen.prompts == [] and check.prompts == []
    assert answer.usage.model_calls == 0


def test_weak_retrieval_refuses_before_generation(corpus, monkeypatch):
    pipeline, gen, _ = build_pipeline(corpus, [])
    monkeypatch.setattr(pipeline.impl, "should_refuse", lambda evidence: True)
    answer = pipeline.ask(HIGH_RISK_Q)
    assert answer.refused and answer.refusal_reason == "weak_retrieval" and gen.prompts == []


def test_unsupported_sentence_is_dropped_and_flagged(corpus):
    pipeline, _, check = build_pipeline(corpus, [ANSWER], ["[1, 0]"])
    answer = pipeline.ask(HIGH_RISK_Q)

    assert answer.answer == "AI systems listed in Annex III are high-risk [AI Act Art. 6(2)]."
    assert answer.removed_sentences == ["Social scoring is banned [AI Act Art. 5(1)]."]
    assert not answer.grounded and answer.grounding_check == "removed_sentences"
    assert [c.label for c in answer.citations] == ["AI Act Art. 6(2)"]  # only what survived


def test_a_citation_that_is_not_among_the_sources_is_removed_without_asking_the_checker(corpus):
    answer_text = "High-risk rules exist [AI Act Art. 6(2)]. Fines are huge [AI Act Art. 99(3)]."
    pipeline, _, check = build_pipeline(corpus, [answer_text], ["[1]"])
    answer = pipeline.ask(HIGH_RISK_Q)

    assert answer.removed_sentences == ["Fines are huge [AI Act Art. 99(3)]."]
    assert "Fines are huge" not in check.prompts[0]  # never sent: the citation is invalid
    assert not answer.grounded


def test_an_unparsable_citation_is_removed(corpus):
    pipeline, _, _ = build_pipeline(
        corpus, ["Good [AI Act Art. 6(2)]. Bad [AI Act Art. ???]."], ["[1]"]
    )
    answer = pipeline.ask(HIGH_RISK_Q)
    assert answer.removed_sentences == ["Bad [AI Act Art. ???]."]


def test_everything_removed_becomes_a_refusal(corpus):
    pipeline, _, _ = build_pipeline(corpus, [ANSWER], ["[0, 0]"])
    answer = pipeline.ask(HIGH_RISK_Q)
    assert answer.refused and answer.refusal_reason == "ungrounded"
    assert len(answer.removed_sentences) == 2 and answer.citations == []


def test_checker_down_keeps_cited_sentences_but_is_not_grounded(corpus):
    text = "A cited claim [AI Act Art. 6(2)]. An uncited claim."
    pipeline, _, _ = build_pipeline(corpus, [text], [RuntimeError("checker down")])
    answer = pipeline.ask(HIGH_RISK_Q)

    assert answer.grounding_check == "unavailable" and not answer.grounded
    assert answer.answer == "A cited claim [AI Act Art. 6(2)]."
    assert answer.removed_sentences == ["An uncited claim."]


def test_checker_with_unreadable_output_is_treated_as_unavailable(corpus):
    pipeline, _, _ = build_pipeline(corpus, [ANSWER], ["I think it is fine"])
    answer = pipeline.ask(HIGH_RISK_Q)
    assert answer.grounding_check == "unavailable" and not answer.grounded


def test_streaming_yields_sources_then_deltas_then_one_final(corpus):
    pipeline, _, _ = build_pipeline(corpus, [ANSWER], ["[1, 1]"])
    events = list(pipeline.events(HIGH_RISK_Q, None, stream=True))

    kinds = [kind for kind, _ in events]
    assert kinds[0] == "sources" and kinds[-1] == "final" and kinds.count("final") == 1
    assert kinds.count("delta") > 1
    assert "".join(p for k, p in events if k == "delta") == ANSWER
    assert events[0][1][0]["label"].startswith("AI Act")


def test_parse_verdicts():
    assert parse_verdicts("[1, 0, 1]", 3) == [True, False, True]
    assert parse_verdicts("Result: [1,0]", 2) == [True, False]
    assert parse_verdicts("[1, 0]", 3) is None  # wrong length
    assert parse_verdicts("[1, 2]", 2) is None  # not 0 or 1
    assert parse_verdicts("fine", 1) is None
