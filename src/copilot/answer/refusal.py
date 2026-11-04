"""When the texts should not be used to answer: the refusal decision."""

from __future__ import annotations

from copilot.answer.evidence import Evidence

# Per language, the best-hit cosine below which the question is refused before any model call.
# STILL PROVISIONAL in all three languages. Measured with `python -m copilot.answer.calibrate`
# on 2026-11-04 (best cosine of the top vector hit, 52 questions):
#
#   en  in-scope n=13: 0.699 to 0.814   out-of-scope n=13: 0.529 to 0.708   -> groups OVERLAP
#   fr  in-scope n=12: 0.728 to 0.835   out-of-scope n=1:  0.576            -> 1 negative only
#   de  in-scope n=12: 0.725 to 0.850   out-of-scope n=1:  0.568            -> 1 negative only
#
# English does not separate: two "adjacent" questions about other laws (the US Executive Order on
# AI, 0.701, and the Digital Services Act, 0.708) score above the lowest in-scope question (0.699).
# No threshold refuses them without also refusing that one, and a wrong refusal is the costlier
# error here, so 0.65 is kept: it refuses no in-scope question and 9 of the 13 out-of-scope ones
# (all 7 unrelated ones and 2 of the 6 adjacent). The other 4 go on to the model's NOT_COVERED
# answer and the grounding check. French and German now get the same 0.65: it refuses no in-scope
# question (lowest 0.728 and 0.725) and their single out-of-scope question, and it is near the
# midpoint of each language's gap (0.652 and 0.647), but one negative per language is not a
# calibration. Languages with no entry here are not gated.
COSINE_MIN: dict[str, float] = {"en": 0.65, "fr": 0.65, "de": 0.65}


def should_refuse(evidence: Evidence) -> bool:
    """Decide from retrieval alone whether to answer "the texts do not cover this".

    Returns True to refuse before any model call, False to go on and generate an answer.

    * No sources at all is always a refusal.
    * In hybrid mode, refuse when the best vector hit is less similar to the question than the
      language's threshold. BM25 scores are NOT used: measured on the gold set, out-of-scope
      questions score as high per token as in-scope ones (the GPU question shares words like
      "training" and "model" with the AI Act).
    * In bm25 mode there is no cosine, so nothing more is refused here; the model's NOT_COVERED
      answer and the grounding check decide. A wrong refusal costs a missed answer, a wrong answer
      costs a false legal statement, so this gate only refuses when fairly sure.
    """
    if evidence.n_sources == 0:
        return True
    threshold = COSINE_MIN.get(evidence.language)
    if threshold is not None and evidence.top_cosine is not None:
        return evidence.top_cosine < threshold
    return False
