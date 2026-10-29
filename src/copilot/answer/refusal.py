"""When the texts should not be used to answer: the refusal decision."""

from __future__ import annotations

from copilot.answer.evidence import Evidence


def should_refuse(evidence: Evidence) -> bool:
    """Decide from retrieval alone whether to answer "the texts do not cover this".

    Return True to refuse before any model call, False to go on and generate an answer.

    TODO(me): implement this, with a threshold you can defend.
    Hints:
      * No sources at all (`n_sources == 0`) is always a refusal.
      * `top_cosine` (hybrid mode) is the signal worth thresholding: how close is the best chunk
        to the question? Look at `python -m copilot.answer.calibrate` for the real values of the
        in-scope and out-of-scope questions before you choose a number, and write down how many
        questions the choice rests on.
      * `top_bm25` is NOT a reliable signal here, measured on the gold set: out-of-scope
        questions score as high as in-scope ones (the GPU question shares words like "training"
        and "model" with the AI Act). Think about what to do in "bm25" mode, where no cosine
        exists: refuse less and lean on the later safeguards (the model's NOT_COVERED answer and
        the grounding check), or refuse more and lose real answers?
      * A wrong refusal costs a missed answer; a wrong answer costs a false legal statement.
        Which error do you want to be rarer?
    The placeholder below only refuses when nothing was retrieved.
    """
    return evidence.n_sources == 0
