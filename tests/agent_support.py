"""Shared fakes for the agent tests: the real corpus, a BM25-only retriever, scripted models."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path

from copilot.agent.context import Models, RunContext
from copilot.agent.graph import AgentRunner, sqlite_checkpointer
from copilot.agent.tools import Tools
from copilot.answer.retrieval import AppRetriever
from copilot.ingest.corpus import read_corpus
from copilot.llm.base import Generation, Usage
from copilot.retrieve.bm25 import BM25Index
from copilot.retrieve.chunks import Corpus
from copilot.retrieve.impl import get_impl
from copilot.retrieve.vector import VectorIndex


class FakeLLM:
    """Answers with fn(prompt); an Exception instance returned by fn is raised instead."""

    def __init__(self, fn: Callable[[str], object], log: list) -> None:
        self.fn, self.log, self.usage = fn, log, []

    def generate(self, prompt, **kwargs):
        self.log.append(prompt)
        out = self.fn(prompt)
        if isinstance(out, Exception):
            raise out
        self.usage.append(Usage("fake", "fake-model", "generate", 100, 20, 0.01))
        return Generation(str(out), "fake-model", 100, 20)


@cache
def units():
    return read_corpus()


@cache
def retriever() -> AppRetriever:
    corpus = Corpus(units())
    return AppRetriever(
        corpus, BM25Index(corpus), VectorIndex(corpus, {}), None, get_impl().rrf_fuse
    )


def labels_in(prompt: str) -> list[str]:
    """Source labels of a Stage 3 answer prompt ("[1] AI Act Art. 5(1) (title)")."""
    return [m.strip() for m in re.findall(r"^\[\d+\] ((?:AI Act|GDPR) [^\n]+?)(?: \([^)]*\))?$",
                                          prompt, re.M)]  # fmt: skip


def cite_first(prompt: str) -> str:
    found = labels_in(prompt)
    return f"The texts say so [{found[0]}]." if found else "NOT_COVERED"


def approve_all(prompt: str) -> str:
    n = len(re.findall(r"^\d+\. ", prompt.split("Sentences to check:")[-1], re.M))
    return "[" + ",".join(["1"] * n) + "]"


def router_says(route: str, confidence: float = 0.9, regulation: str = "aiact", question=None):
    return json.dumps({"route": route, "confidence": confidence, "regulation": regulation,
                       "clarifying_question": question})  # fmt: skip


class Script:
    """Per-purpose behaviour for the fake models, with call logs."""

    def __init__(self, **fns: Callable[[str], object]) -> None:
        self.logs: dict[str, list[str]] = {}
        defaults = {
            "router": lambda p: router_says("legal_question"),
            "rewrite": lambda p: (
                '{"queries": [{"regulation": "aiact", "query": "prohibited AI practices"}]}'
            ),
            "grade": lambda p: '{"relevant": [1, 2], "decision": "answer", "reason": "ok"}',
            "react": lambda p: '{"done": "nothing to do"}',
            "risk": lambda p: '{"facts": {}, "annex_iii_area": "none", "roles": ["provider"]}',
            "answer": cite_first,
            "report": cite_first,
            "ground": approve_all,
        }
        self.fns = {**defaults, **fns}

    def factory(self, purpose: str):
        log = self.logs.setdefault(purpose, [])
        return lambda: FakeLLM(self.fns[purpose], log)

    def models(self) -> Models:
        names = ["router", "rewrite", "grade", "react", "answer", "report", "ground"]
        return Models({name: self.factory(name) for name in names})


def build_runner(script: Script, checkpoints: Path | str = ":memory:") -> AgentRunner:
    tools = Tools(units(), retriever(), risk_llm_factory=script.factory("risk"))
    return AgentRunner(RunContext(tools, script.models()), sqlite_checkpointer(Path(checkpoints)))


def tools_for(script: Script | None = None) -> Tools:
    risk = (script or Script()).factory("risk")
    return Tools(units(), retriever(), risk_llm_factory=risk)
