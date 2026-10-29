"""Request and response shapes shared by the pipeline and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["en", "fr", "de"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    language: Language | None = Field(
        default=None, description="Language of the question and answer; detected if omitted."
    )


class Citation(BaseModel):
    id: str  # corpus unit id, e.g. "aiact:art:6:2"
    label: str  # as cited in the answer, e.g. "AI Act Art. 6(2)"
    regulation: Literal["aiact", "gdpr"]
    snippet: str  # start of the cited text


class Usage(BaseModel):
    """What the request used. Token counts are None when a provider did not report them."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    model_calls: int = 0
    models: list[str] = []  # "provider:model" of each successful call
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    total_ms: float = 0.0


class Answer(BaseModel):
    answer: str
    language: Language
    refused: bool  # True: "the texts do not cover this" (see refusal_reason)
    refusal_reason: Literal["no_sources", "weak_retrieval", "not_covered", "ungrounded"] | None = (
        None
    )
    citations: list[Citation]
    grounded: bool  # every sentence was checked against its cited sources and none was removed
    grounding_check: Literal["passed", "removed_sentences", "unavailable", "skipped"]
    removed_sentences: list[str] = []
    retrieval_mode: Literal["hybrid", "bm25"]
    retrieval_notes: list[str] = []
    usage: Usage
