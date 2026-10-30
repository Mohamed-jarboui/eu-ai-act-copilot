"""Input and output guardrails (docs/agent-design.md, section 5). Action guards live in context.py.

Cheap rules first. The input guard blocks only what must never run (attempts to extract the
prompt, keys or configuration, or to change the assistant's behaviour); a legal question that
merely contains injected text continues with the suspicious span removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_CHARS, MAX_CHARS = 3, 2000
MAX_ANSWER_WORDS = {"default": 600, "compliance_check": 900}

# Phrases that try to take over the assistant. Matched case-insensitively, in EN/FR/DE.
_INJECTION = [
    r"ignore (?:all |the |any )?(?:previous|prior|above|earlier) (?:instructions|prompts?|rules)",
    r"disregard (?:all |the |any )?(?:previous|prior|above) (?:instructions|rules)",
    r"(?:forget|override) (?:your|all|the) (?:instructions|rules|guidelines)",
    r"you are now (?:a|an|in) ",
    r"(?:act|pretend|behave) as (?:if you were |an? )?"
    r"(?:unrestricted|jailbroken|dan\b|developer mode)",
    r"new (?:system )?instructions?:",
    r"ignore(?:z)? (?:les|toutes les) instructions (?:précédentes|ci-dessus)",
    r"ignoriere (?:alle |die )?(?:vorherigen|bisherigen|obigen) (?:anweisungen|regeln)",
    r"<\s*/?\s*system\s*>",
]
# Requests that must be blocked outright: they target the system, not the law.
_EXFILTRATION = [
    r"(?:reveal|show|print|repeat|give me|tell me|what (?:is|are)) (?:me )?(?:your|the) "
    r"(?:system prompt|prompt|instructions|api key|keys?|configuration|config|secrets?)",
    r"(?:montre|affiche|révèle|donne)[- ](?:moi )?(?:ton|tes|le|les|votre|vos) "
    r"(?:prompt|instructions|clés?|configuration)",
    r"(?:zeige|verrate|nenne) (?:mir )?(?:deinen|deine|den|die|ihren|ihre) "
    r"(?:prompt|systemprompt|anweisungen|schlüssel|konfiguration)",
    r"\.env\b|api[_ -]?key",
]
_ENCODED_BLOB = r"[A-Za-z0-9+/]{120,}={0,2}"
_PII = [
    (r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]"),
    (r"\+?\d[\d ()./-]{7,}\d", "[number]"),
]


@dataclass
class InputVerdict:
    blocked: bool
    reason: str | None = None
    flagged: bool = False  # injected text found and removed, the question continues
    sanitized: str = ""
    removed: list[str] = field(default_factory=list)


def check_input(question: str) -> InputVerdict:
    text = question.strip()
    if not MIN_CHARS <= len(text) <= MAX_CHARS:
        return InputVerdict(True, "length", sanitized=text)
    lowered = text.lower()
    for pattern in _EXFILTRATION:
        if re.search(pattern, lowered):
            return InputVerdict(True, "exfiltration", sanitized=text)
    removed: list[str] = []
    sanitized = text
    for pattern in [*_INJECTION, _ENCODED_BLOB]:
        for match in re.finditer(pattern, sanitized, re.IGNORECASE):
            removed.append(match.group(0))
        sanitized = re.sub(pattern, " ", sanitized, flags=re.IGNORECASE)
    sanitized = " ".join(sanitized.split())
    if removed and len(sanitized) < MIN_CHARS:
        return InputVerdict(True, "injection", sanitized=sanitized, removed=removed)
    return InputVerdict(False, flagged=bool(removed), sanitized=sanitized, removed=removed)


def redact(text: str) -> str:
    """Personal data out of logs and traces (the question itself is kept for answering)."""
    for pattern, token in _PII:
        text = re.sub(pattern, token, text)
    return text


BLOCKED_MESSAGES = {
    "en": "I can only answer questions about the AI Act and the GDPR, and I cannot share my "
    "instructions or configuration.",
    "fr": "Je ne peux répondre qu'à des questions sur le règlement sur l'IA et le RGPD, et je ne "
    "peux pas communiquer mes instructions ni ma configuration.",
    "de": "Ich kann nur Fragen zur KI-Verordnung und zur DSGVO beantworten und gebe meine "
    "Anweisungen oder Konfiguration nicht weiter.",
}
NOTICE = {
    "en": "This is information about the texts, not legal advice.",
    "fr": "Il s'agit d'informations sur les textes, pas d'un conseil juridique.",
    "de": "Dies sind Informationen über die Texte, keine Rechtsberatung.",
}


def check_output(answer: str, route: str | None, cited: int, refused: bool) -> list[str]:
    """Problems with a final answer (empty list = fine). The caller decides what to do."""
    problems = []
    words = len(answer.split())
    limit = MAX_ANSWER_WORDS.get(route or "", MAX_ANSWER_WORDS["default"])
    if words > limit:
        problems.append(f"too_long:{words}>{limit}")
    if re.search(r"<[a-z/][^>]*>|https?://", answer, re.IGNORECASE):
        problems.append("html_or_link")
    if not refused and cited == 0:
        problems.append("no_citations")
    return problems
