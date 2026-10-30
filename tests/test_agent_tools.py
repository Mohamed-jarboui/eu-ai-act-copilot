"""The six tools: schemas, recovery errors, bounded outputs, and the risk rules."""

import json

import pytest

from agent_support import Script, tools_for
from copilot.agent import risk
from copilot.agent.tools import ARG_MODELS, DESCRIPTIONS, run_tool, tool_catalog


@pytest.fixture(scope="module")
def tools():
    return tools_for()


def call(tools, name, **args):
    return run_tool(tools, name, args)


# --- schemas and descriptions --------------------------------------------------------------


def test_every_tool_has_a_prompt_description_and_a_schema():
    assert set(DESCRIPTIONS) == set(ARG_MODELS) == {
        "search_regulation", "get_article", "get_definitions", "classify_ai_risk",
        "list_obligations", "cross_reference",
    }  # fmt: skip
    for name, text in DESCRIPTIONS.items():
        lowered = text.lower()  # each says when to use it, and when not
        assert "use it" in lowered and ("do not" in lowered or "never" in lowered), name
    catalog = tool_catalog(["get_article"])
    assert "get_article" in catalog and '"reference"' in catalog


@pytest.mark.parametrize(
    ("name", "args", "hint"),
    [
        ("search_regulation", {"query": "ai", "language": "en"}, "query"),
        ("search_regulation", {"query": "prohibited practices", "language": "es"}, "language"),
        ("search_regulation", {"query": "prohibited practices", "language": "en", "k": 20}, "k"),
        ("get_article", {"regulation": "dsa", "reference": "5", "language": "en"}, "regulation"),
        ("get_definitions", {"term": "x", "language": "en"}, "term"),
        ("classify_ai_risk", {"system_description": "a bot", "language": "en"}, "purpose"),
        ("list_obligations", {"role": "user", "category": "high_risk", "language": "en"},
         "provider, deployer"),
        ("list_obligations", {"role": "provider", "category": "prohibited", "language": "en"},
         "prohibited systems have no obligations"),
        ("cross_reference", {"regulation": "aiact", "language": "en"}, "reference"),
    ],
)  # fmt: skip
def test_invalid_arguments_return_a_recovery_message(tools, name, args, hint):
    result = run_tool(tools, name, args)
    assert not result.ok and result.error.error == "invalid_argument"
    assert result.error.message.startswith(f"Invalid arguments for {name}")
    assert hint in result.error.message


def test_unknown_tool(tools):
    result = run_tool(tools, "send_email", {})
    assert result.error.error == "not_allowed" and "search_regulation" in result.error.message


# --- search_regulation ----------------------------------------------------------------------


def test_search_returns_bounded_labelled_snippets(tools):
    result = call(tools, "search_regulation", query="prohibited AI practices", language="en", k=4)
    assert result.ok and len(result.output["results"]) == 4
    for item in result.output["results"]:
        assert item["label"].startswith(("AI Act", "GDPR")) and len(item["snippet"]) <= 601
    assert result.output["mode"] == "bm25" and len(result.sources) == 4


def test_search_filters_by_regulation(tools):
    result = call(tools, "search_regulation", query="personal data breach notification",
                  regulation="gdpr", language="en")  # fmt: skip
    assert all(r["label"].startswith("GDPR") for r in result.output["results"])


def test_search_with_no_match_tells_the_model_what_to_try(tools):
    result = call(tools, "search_regulation", query="zzqx wwqv yyqk", language="en")
    assert result.error.error == "no_results" and "Rephrase" in result.error.message


# --- get_article ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("regulation", "reference", "label"),
    [
        ("gdpr", "22", "GDPR Art. 22"),
        ("gdpr", "Article 22", "GDPR Art. 22"),
        ("aiact", "6(2)", "AI Act Art. 6(2)"),
        ("aiact", "5(1)(f)", "AI Act Art. 5(1)(f)"),
        ("aiact", "Annex III, point 4", "AI Act Annex III, point 4"),
        ("aiact", "3(1)", "AI Act Art. 3(1)"),  # a definition, cited as its article
    ],
)
def test_get_article_accepts_the_usual_reference_forms(tools, regulation, reference, label):
    result = call(tools, "get_article", regulation=regulation, reference=reference, language="en")
    assert result.ok and result.output["label"] == label
    assert result.sources[0].label == label


def test_get_article_bounds_long_text_and_lists_parts(tools):
    result = call(tools, "get_article", regulation="aiact", reference="5(1)", language="en")
    assert result.output["truncated"] is True and len(result.output["text"]) <= 4001
    assert "AI Act Art. 5(1)(a)" in result.output["parts"]


@pytest.mark.parametrize(
    ("regulation", "reference", "kind"),
    [
        ("aiact", "114", "not_found"),
        ("gdpr", "100", "not_found"),
        ("gdpr", "Annex I", "not_found"),  # the GDPR has no annexes
        ("aiact", "Annex XIV", "not_found"),
        ("aiact", "6(9)", "not_found"),
        ("aiact", "six", "invalid_argument"),
    ],
)
def test_get_article_errors(tools, regulation, reference, kind):
    result = call(tools, "get_article", regulation=regulation, reference=reference, language="en")
    assert result.error.error == kind
    if kind == "not_found" and "(" not in reference:
        assert "1 to 113" in result.error.message and "1 to 99" in result.error.message


def test_get_article_in_french(tools):
    result = call(tools, "get_article", regulation="gdpr", reference="22", language="fr")
    assert result.ok and "décision" in result.output["text"].lower()


# --- get_definitions ------------------------------------------------------------------------


def test_definition_exact_match(tools):
    result = call(tools, "get_definitions", term="AI system", language="en")
    first = result.output["definitions"][0]
    assert first["label"] == "AI Act Art. 3(1)" and first["term"] == "AI system"


def test_definition_ignores_case_and_accents(tools):
    result = call(tools, "get_definitions", term="donnees a caractere personnel", language="fr",
                  regulation="gdpr")  # fmt: skip
    assert result.output["definitions"][0]["label"] == "GDPR Art. 4(1)"


def test_definition_no_match_suggests_close_terms(tools):
    result = call(tools, "get_definitions", term="deployr", regulation="aiact", language="en")
    assert result.ok  # fuzzy match on the term
    missing = call(tools, "get_definitions", term="blockchain", language="en")
    assert missing.error.error == "no_results" and "Closest defined terms" in missing.error.message


# --- classify_ai_risk: the rules ------------------------------------------------------------


def facts(area="none", roles=("provider",), **values):
    return risk.RiskFacts(
        facts={k: risk.Fact(value=v) for k, v in values.items()}, annex_iii_area=area,
        roles=list(roles),
    )  # fmt: skip


def no_unknowns(**values):
    """All checklist facts "no" unless given, so only the rule under test decides."""
    return {name: "no" for name in risk.fact_names()} | values


def test_rule_prohibited_practice():
    result = risk.decide(facts(**no_unknowns(art5_1_f="yes")), "en")
    assert result.category == "prohibited" and result.basis == ["aiact:art:5:1:f"]


def test_rule_annex_iii_is_high_risk():
    result = risk.decide(facts("4", **no_unknowns(art6_3_derogation="no")), "en")
    assert result.category == "high_risk"
    assert result.basis[:2] == ["aiact:art:6:2", "aiact:anx:III:4"]


def test_rule_derogation_applies_without_profiling():
    result = risk.decide(facts("4", **no_unknowns(art6_3_derogation="yes")), "en")
    assert result.category == "minimal" and "aiact:art:6:3" in result.basis
    assert result.derogation_considered and result.assumptions


def test_rule_profiling_cancels_the_derogation():
    result = risk.decide(facts("4", **no_unknowns(art6_3_derogation="yes", profiling="yes")), "en")
    assert result.category == "high_risk" and "aiact:art:6:3" in result.basis


def test_rule_unknown_derogation_assumes_high_risk_and_asks():
    result = risk.decide(facts("5", **no_unknowns(art6_3_derogation="unknown")), "fr")
    assert result.category == "high_risk"
    assert any("procédurale" in q for q in result.missing_facts) and result.assumptions


def test_rule_unknown_area_is_undetermined_with_a_question():
    result = risk.decide(facts("unknown", **no_unknowns()), "de")
    assert result.category == "undetermined" and any("Bereich" in q for q in result.missing_facts)


def test_rule_chatbot_is_limited_with_transparency():
    result = risk.decide(facts("none", **no_unknowns(art50_interacts="yes")), "en")
    assert result.category == "limited" and result.transparency_obligations
    assert result.basis == ["aiact:art:50:1"]


def test_rule_high_risk_can_also_have_transparency_duties():
    result = risk.decide(facts("1", **no_unknowns(art50_emotion_biometric="yes")), "en")
    assert result.category == "high_risk" and result.transparency_obligations


def test_rule_safety_component_and_gpai():
    assert risk.decide(facts(**no_unknowns(annex_i_safety_component="yes")), "en").basis == [
        "aiact:art:6:1"
    ]
    result = risk.decide(facts(**no_unknowns(gpai_model="yes")), "en")
    assert result.gpai and "aiact:art:53" in result.basis


def test_rule_missing_role_becomes_a_question():
    result = risk.decide(facts("none", roles=(), **no_unknowns()), "en")
    assert any("provider" in q for q in result.missing_facts)


def test_parse_facts_ignores_unknown_keys_and_bad_areas():
    parsed = risk.parse_facts(
        '{"facts": {"made_up": {"value": "yes"}, "profiling": {"value": '
        '"yes"}}, "annex_iii_area": "12", "roles": ["deployer"]}'
    )
    assert set(parsed.facts) == {"profiling"} and parsed.annex_iii_area == "unknown"
    assert risk.parse_facts("not json") is None


# --- classify_ai_risk: the tool -------------------------------------------------------------

CV_TOOL = "We build a CV-screening tool that ranks job applicants for recruiters at our clients."


def test_classify_tool_runs_the_extraction_then_the_rules():
    extraction = {"facts": {"profiling": {"value": "yes", "evidence": "ranks"}},
                  "annex_iii_area": "4", "roles": ["provider"]}  # fmt: skip
    tools = tools_for(Script(risk=lambda p: json.dumps(extraction)))
    result = call(tools, "classify_ai_risk", system_description=CV_TOOL, language="en")
    assert result.ok and result.output["category"] == "high_risk"
    assert "AI Act Annex III, point 4" in result.output["basis"]
    assert {s.label for s in result.sources} >= {"AI Act Art. 6(2)", "AI Act Annex III, point 4"}
    assert result.llm is not None  # its model call is charged to the run's budget


def test_classify_tool_errors():
    tools = tools_for(
        Script(risk=lambda p: '{"facts": {}, "annex_iii_area": "unknown", "roles": []}')
    )
    vague = call(tools, "classify_ai_risk", system_description="It is an AI thing that we have.",
                 language="en")  # fmt: skip
    assert vague.error.error == "insufficient_facts" and "purpose" in vague.error.message
    long = call(tools, "classify_ai_risk", system_description="x" * 3001, language="en")
    assert long.error.error == "too_long"
    broken = call(tools_for(Script(risk=lambda p: "sorry")), "classify_ai_risk",
                  system_description=CV_TOOL, language="en")  # fmt: skip
    assert broken.error.error == "insufficient_facts" and "unreadable twice" in broken.error.message


def test_known_facts_override_the_extraction():
    tools = tools_for(Script(risk=lambda p: '{"facts": {}, "annex_iii_area": "none", "roles": []}'))
    known = {"facts": {}, "annex_iii_area": "4", "roles": ["deployer"]}
    result = call(tools, "classify_ai_risk", system_description=CV_TOOL, known_facts=known,
                  language="en")  # fmt: skip
    assert result.output["category"] == "high_risk" and result.output["roles"] == ["deployer"]


# --- list_obligations -------------------------------------------------------------------------


def test_provider_high_risk_obligations_are_art_16_quotes(tools):
    result = call(tools, "list_obligations", role="provider", category="high_risk", language="en")
    labels = [o["label"] for o in result.output["obligations"]]
    assert labels[0] == "AI Act Art. 16(a)" and len(labels) == 12
    assert all(len(o["quote"]) <= 301 for o in result.output["obligations"])


def test_deployer_high_risk_lists_art_26_and_27(tools):
    result = call(tools, "list_obligations", role="deployer", category="high_risk", language="en")
    labels = [o["label"] for o in result.output["obligations"]]
    assert (
        len(labels) == 17 and labels[0] == "AI Act Art. 26(1)" and labels[-1] == "AI Act Art. 27(5)"
    )
    assert result.output["more"] is False


def test_obligations_are_capped_at_20(tools, tmp_path, monkeypatch):
    from copilot.agent import tools as tools_module

    table = {"provider|high_risk": ["aiact:art:16", "aiact:art:26", "aiact:art:27"]}  # 29 parts
    (tmp_path / "obligations.json").write_text(json.dumps(table), "utf-8")
    monkeypatch.setattr(tools_module, "DATA", tmp_path)
    result = call(tools, "list_obligations", role="provider", category="high_risk", language="en")
    assert len(result.output["obligations"]) == 20 and result.output["more"] is True


def test_limited_risk_obligations_depend_on_the_role(tools):
    deployer = call(tools, "list_obligations", role="deployer", category="limited", language="de")
    assert [o["label"] for o in deployer.output["obligations"]] == [
        "AI Act Art. 50(3)", "AI Act Art. 50(4)"
    ]  # fmt: skip
    missing = call(tools, "list_obligations", role="importer", category="limited", language="en")
    assert missing.error.error == "not_found" and "banned" in missing.error.message


# --- cross_reference --------------------------------------------------------------------------


def test_cross_reference_finds_explicit_gdpr_links_from_the_text(tools):
    result = call(tools, "cross_reference", regulation="aiact", reference="26", language="en")
    other = {o["label"]: o["basis"] for o in result.output["other_regulation"]}
    assert other["GDPR Art. 35"].startswith("explicit reference in AI Act Art. 26(9)")
    assert "GDPR Art. 9" in other
    assert len(result.output["refers_to"]) <= 10 and len(result.output["referred_by"]) <= 10


def test_cross_reference_from_the_gdpr_side_includes_curated_pairs(tools):
    result = call(tools, "cross_reference", regulation="gdpr", reference="22", language="en")
    other = {o["label"]: o["basis"] for o in result.output["other_regulation"]}
    assert other["AI Act Art. 86"].startswith("curated:")
    assert result.output["referred_by"]  # other GDPR articles refer to Art. 22
