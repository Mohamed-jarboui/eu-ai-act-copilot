"""`python -m copilot.answer.examples`: run 10 hand-picked questions and put the real answers in the
README (between the examples markers). Needs the API keys; takes a couple of minutes."""

from __future__ import annotations

from pathlib import Path

from copilot.answer.factory import build_pipeline
from copilot.answer.models import Answer

README = Path("README.md")
START, END = "<!-- examples:start -->", "<!-- examples:end -->"

# (language, question). Hand-picked from the retrieval gold set to cover both regulations, all
# three languages, an annex, a definition, a question needing both regulations, an out-of-scope
# question, and one question where the system is known to miss.
QUESTIONS = [
    ("en", "Which uses of AI are completely banned in the EU, whoever deploys them?"),
    ("en", "Is an AI tool that ranks job applicants considered high-risk?"),
    ("en", "Within how many hours must a data breach be reported to the authority?"),
    (
        "en",
        "A bank's AI model refuses loans automatically. What can the rejected applicant demand?",
    ),
    ("en", "What is the best GPU for training a large language model?"),
    ("fr", "Comment le règlement définit-il un système d'IA ?"),
    ("fr", "Un employeur peut-il utiliser l'IA pour détecter les émotions de ses salariés ?"),
    ("fr", "Quand un consentement est-il valable ?"),
    ("de", "Müssen Deepfakes gekennzeichnet werden?"),
    ("de", "Wann darf ich verlangen, dass meine Daten gelöscht werden?"),
]


def render(index: int, question: str, a: Answer) -> str:
    u = a.usage
    tokens = (
        "tokens not reported"
        if u.input_tokens is None
        else f"{u.input_tokens} in / {u.output_tokens} out tokens"
    )
    status = (
        f"no answer ({a.refusal_reason})"
        if a.refused
        else ("grounded" if a.grounded else f"NOT fully grounded ({a.grounding_check})")
    )
    quoted = "\n".join(f"> {line}" if line else ">" for line in a.answer.split("\n"))
    lines = [
        f"**{index}. [{a.language.upper()}] {question}**",
        "",
        quoted,
        "",
        f"Citations: {', '.join(c.label for c in a.citations) or 'none'}  ",
        f"Status: {status} · retrieval: {a.retrieval_mode} · {u.total_ms / 1000:.1f} s · "
        f"{tokens} · models: {', '.join(u.models) or 'none'}",
    ]
    if a.removed_sentences:
        lines.append(f"  \nRemoved as unsupported: {' | '.join(a.removed_sentences)}")
    return "\n".join(lines)


def main() -> None:
    pipeline = build_pipeline()
    modes = pipeline.retriever.modes()
    blocks = []
    for number, (language, question) in enumerate(QUESTIONS, start=1):
        print(f"[{number}/{len(QUESTIONS)}] {question}", flush=True)
        try:
            blocks.append(render(number, question, pipeline.ask(question, language)))
        except Exception as exc:  # keep the other answers: report this failure as it happened
            blocks.append(
                f"**{number}. [{language.upper()}] {question}**\n\n"
                f"> The request failed: `{type(exc).__name__}`: {str(exc)[:200]}"
            )
    with_vectors = ", ".join(lang for lang, mode in modes.items() if mode == "hybrid")
    body = (
        "Real output of `python -m copilot.answer.examples`, unedited. "
        f"Vectors were available for: {with_vectors or 'no language'}; every other language, and "
        "any question whose embedding could not be fetched, used BM25 only (see each "
        "`retrieval:`).\n\n" + "\n\n".join(blocks)
    )
    text = README.read_text("utf-8")
    start, end = text.index(START) + len(START), text.index(END)
    README.write_text(text[:start] + "\n" + body + "\n" + text[end:], "utf-8")
    print("README updated")


if __name__ == "__main__":
    main()
