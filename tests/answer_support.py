"""Shared fakes for the answer and API tests."""

from copilot.answer.impl import get_impl
from copilot.answer.pipeline import AnswerPipeline
from copilot.answer.retrieval import AppRetriever
from copilot.llm.base import Generation, StreamEvent, Usage
from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.vector import VectorIndex

HIGH_RISK_Q = "Which AI systems are high-risk, and which AI practices are prohibited?"
FR_Q = "Quels systèmes d'IA sont à haut risque et quelles pratiques sont interdites ?"
DE_Q = "Welche KI-Systeme sind hochriskant und welche Praktiken sind verboten?"
ANSWER = (
    "AI systems listed in Annex III are high-risk [AI Act Art. 6(2)]. "
    "Social scoring is banned [AI Act Art. 5(1)]."
)


class FakeLLM:
    """Plays back scripted outputs (an Exception instance is raised) and records usage."""

    def __init__(self, outputs: list) -> None:
        self.outputs, self.prompts, self.usage = list(outputs), [], []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        self.usage.append(Usage("fake", "m", "generate", 100, 10, 0.01))
        return Generation(out, "m", 100, 10)

    def generate_stream(self, prompt, **kwargs):
        text = self.generate(prompt, **kwargs).text
        for i in range(0, len(text), 25):
            yield StreamEvent(text=text[i : i + 25])
        yield StreamEvent(final=Generation(text, "m", 100, 10))


def build_pipeline(corpus, gen_outputs, check_outputs=(), impl="mine"):
    gen, check = FakeLLM(gen_outputs), FakeLLM(check_outputs)
    retriever = AppRetriever(
        corpus, BM25Index(corpus), VectorIndex(corpus, {}), None, lambda rankings: []
    )
    pipeline = AnswerPipeline(retriever, lambda: gen, lambda: check, get_impl(impl), k=4)
    return pipeline, gen, check
