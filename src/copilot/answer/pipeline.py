"""retrieve -> refusal gate -> generate -> citation analysis -> grounding check -> answer."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

from copilot.answer.grounding import analyse, check_grounding
from copilot.answer.impl import Impl
from copilot.answer.labels import cite_id, label_for
from copilot.answer.language import detect_language
from copilot.answer.models import Answer, Citation, Usage
from copilot.answer.prompt import NOT_COVERED, REFUSALS, SYSTEM, Source, build_prompt
from copilot.answer.retrieval import AppRetriever, Retrieved

SNIPPET_CHARS = 300
Event = tuple[str, Any]  # ("sources", [...]) | ("delta", "text") | ("final", Answer)


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + "…"


def _tokens(*llms) -> tuple[int | None, int | None, int, list[str]]:
    """(input tokens, output tokens, successful calls, "provider:model" of each call).

    A token total is None if any call did not report its tokens.
    """
    records = [u for llm in llms if llm for u in llm.usage if u.error is None]
    totals = []
    for attr in ("input_tokens", "output_tokens"):
        values = [getattr(u, attr) for u in records]
        totals.append(None if not records or any(v is None for v in values) else sum(values))
    return totals[0], totals[1], len(records), [f"{u.provider}:{u.model}" for u in records]


class AnswerPipeline:
    def __init__(
        self,
        retriever: AppRetriever,
        make_generation_llm: Callable[[], Any],
        make_check_llm: Callable[[], Any],
        impl: Impl,
        *,
        k: int = 8,
    ) -> None:
        self.retriever = retriever
        self.make_generation_llm = make_generation_llm
        self.make_check_llm = make_check_llm
        self.impl = impl
        self.k = k

    # -- public API ----------------------------------------------------------------------

    def ask(self, question: str, language: str | None = None) -> Answer:
        for kind, payload in self.events(question, language, stream=False):
            if kind == "final":
                return payload
        raise RuntimeError("pipeline ended without an answer")

    def events(
        self, question: str, language: str | None = None, *, stream: bool
    ) -> Iterator[Event]:
        started = time.perf_counter()
        lang = language or detect_language(question)
        retrieved = self.retriever.retrieve(question, lang, self.k)
        sources = self._sources(lang, retrieved.ids)
        yield "sources", [self._source_info(s) for s in sources]

        def refusal(reason: str, **extra) -> Event:
            return "final", self._refusal(lang, reason, retrieved, started, **extra)

        if not sources:
            yield refusal("no_sources")
            return
        if self.impl.should_refuse(retrieved.evidence):
            yield refusal("weak_retrieval")
            return

        gen_llm, check_llm = self.make_generation_llm(), self.make_check_llm()
        prompt = build_prompt(question, sources, lang)
        gen_start = time.perf_counter()
        text = ""
        if stream:
            for event in gen_llm.generate_stream(
                prompt, system=SYSTEM, tier="large", max_output_tokens=3000
            ):
                if event.text:
                    text += event.text
                    yield "delta", event.text
                if event.final is not None:
                    text = event.final.text
        else:
            text = gen_llm.generate(
                prompt, system=SYSTEM, tier="large", max_output_tokens=3000
            ).text
        generation_s = time.perf_counter() - gen_start

        if NOT_COVERED in text:
            yield refusal("not_covered", llms=(gen_llm,), generation_s=generation_s)
            return

        ground_start = time.perf_counter()
        sentences = analyse(text, sources, self.impl.parse_citation)
        grounding = check_grounding(check_llm, sentences, sources)
        grounding_s = time.perf_counter() - ground_start
        if not grounding.kept:
            yield refusal(
                "ungrounded",
                llms=(gen_llm, check_llm),
                generation_s=generation_s,
                grounding_s=grounding_s,
                removed=grounding.removed,
            )
            return

        by_cite = {s.cite_id: s for s in sources}
        citations = [self._citation(by_cite[c]) for c in grounding.verified_cites]
        input_tokens, output_tokens, calls, models = _tokens(gen_llm, check_llm)
        yield (
            "final",
            Answer(
                answer=" ".join(grounding.kept),
                language=lang,
                refused=False,
                citations=citations,
                grounded=grounding.status == "passed",
                grounding_check=grounding.status,
                removed_sentences=grounding.removed,
                retrieval_mode=retrieved.evidence.mode,
                retrieval_notes=retrieved.notes,
                usage=Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_calls=calls,
                    models=models,
                    retrieval_ms=round(retrieved.latency_s * 1000, 1),
                    generation_ms=round(generation_s * 1000, 1),
                    grounding_ms=round(grounding_s * 1000, 1),
                    total_ms=round((time.perf_counter() - started) * 1000, 1),
                ),
            ),
        )

    # -- helpers ---------------------------------------------------------------------------

    def _sources(self, lang: str, chunk_ids: list[str]) -> list[Source]:
        sources: dict[str, Source] = {}
        for chunk_id in chunk_ids:
            unit = self.retriever.by_id[(lang, chunk_id)]
            cid = cite_id(unit)
            sources.setdefault(
                cid, Source(cid, label_for(cid), unit.id, unit.regulation, unit.title, unit.text)
            )
        return list(sources.values())

    @staticmethod
    def _source_info(source: Source) -> dict:
        return {
            "id": source.unit_id,
            "label": source.label,
            "regulation": source.regulation,
            "snippet": _snippet(source.text),
        }

    @staticmethod
    def _citation(source: Source) -> Citation:
        return Citation(
            id=source.unit_id,
            label=source.label,
            regulation=source.regulation,
            snippet=_snippet(source.text),
        )

    def _refusal(
        self,
        lang: str,
        reason: str,
        retrieved: Retrieved,
        started: float,
        *,
        llms: tuple = (),
        generation_s: float = 0.0,
        grounding_s: float = 0.0,
        removed: list[str] | None = None,
    ) -> Answer:
        input_tokens, output_tokens, calls, models = _tokens(*llms)
        return Answer(
            answer=REFUSALS[lang],
            language=lang,
            refused=True,
            refusal_reason=reason,
            citations=[],
            grounded=True,  # nothing is claimed about the law, so nothing is unsupported
            grounding_check="skipped",
            removed_sentences=removed or [],
            retrieval_mode=retrieved.evidence.mode,
            retrieval_notes=retrieved.notes,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_calls=calls,
                models=models,
                retrieval_ms=round(retrieved.latency_s * 1000, 1),
                generation_ms=round(generation_s * 1000, 1),
                grounding_ms=round(grounding_s * 1000, 1),
                total_ms=round((time.perf_counter() - started) * 1000, 1),
            ),
        )
