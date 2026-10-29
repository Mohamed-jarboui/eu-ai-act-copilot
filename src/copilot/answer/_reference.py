"""Reference versions of the TODO(me) functions in citations.py and refusal.py.

Spoiler: these are the answers. Try your own first. The app and the tests can run with either
(see impl.py); yours must pass the same tests, including the round trip over the whole corpus.
"""

from __future__ import annotations

import re

from copilot.answer.evidence import Evidence

_REGULATION = {"ai act": "aiact", "gdpr": "gdpr"}
_ARTICLE = re.compile(
    r"^(AI\s*Act|GDPR)\s+Art(?:icle)?\.?\s*(\d+[a-z]?)\s*((?:\(\s*\w+\s*\)\s*)*)$", re.IGNORECASE
)
_ANNEX = re.compile(
    r"^(AI\s*Act|GDPR)\s+Annex\s+([IVXLC]+)"
    r"(?:\s*,\s*Section\s+(\w+))?(?:\s*,\s*point\s+\(?(\w+)\)?)?$",
    re.IGNORECASE,
)

# Cosine below this means "the best chunk is not about the question". None = gate disabled: it
# cannot be set honestly until out-of-scope questions have cosine scores (see calibrate.py).
COSINE_MIN: float | None = None


def parse_citation(label: str) -> str | None:
    label = " ".join(label.split())
    if match := _ARTICLE.match(label):
        regulation = _REGULATION[re.sub(r"\s+", " ", match.group(1).lower())]
        parts = re.findall(r"\(\s*(\w+)\s*\)", match.group(3))
        return ":".join([regulation, "art", match.group(2).lower(), *[p.lower() for p in parts]])
    if match := _ANNEX.match(label):
        regulation = _REGULATION[re.sub(r"\s+", " ", match.group(1).lower())]
        parts = [match.group(2).upper()]
        if match.group(3):
            parts.append(match.group(3).upper())
        if match.group(4):
            parts.append(match.group(4).lower())
        return ":".join([regulation, "anx", *parts])
    return None


def should_refuse(evidence: Evidence) -> bool:
    if evidence.n_sources == 0:
        return True
    if COSINE_MIN is not None and evidence.top_cosine is not None:
        return evidence.top_cosine < COSINE_MIN
    return False  # bm25 mode, or gate not calibrated: later safeguards decide
