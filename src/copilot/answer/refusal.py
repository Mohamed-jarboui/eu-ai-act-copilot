"""When the texts should not be used to answer: the refusal decision."""

from __future__ import annotations

from copilot.answer.evidence import Evidence

# Per language, the best-hit cosine below which the question is refused before any model call.
# PROVISIONAL and English only: it rests on 9 in-scope English gold questions (best cosine 0.699 to
# 0.814) and 1 out-of-scope one (the GPU question, 0.599); the daily embedding quota prevented
# more. 0.65 sits between the two groups. It is deliberately conservative (nearer the lowest
# in-scope value than a guess in the middle of nothing) because the model's NOT_COVERED answer and
# the grounding check still follow. Recalibrate with `python -m copilot.answer.calibrate --embed`
# once vectors exist for French and German and the out-of-scope set is larger; languages with no
# entry here are not gated.
COSINE_MIN: dict[str, float] = {"en": 0.65}


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
