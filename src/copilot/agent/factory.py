"""Build the real agent from the Stage 2 retriever, the Stage 3 model chains and SQLite."""

from __future__ import annotations

from pathlib import Path

from copilot.agent.context import Models, RunContext
from copilot.agent.graph import CHECKPOINTS, AgentRunner, sqlite_checkpointer
from copilot.agent.tools import Tools
from copilot.answer.factory import build_retriever, model_chains
from copilot.ingest.corpus import read_corpus
from copilot.retrieve.chunks import Corpus


def build_agent(checkpoints: Path = CHECKPOINTS, retriever=None) -> AgentRunner:
    units = read_corpus()
    retriever = retriever or build_retriever(Corpus(units))
    chains = model_chains()
    small, generation, check = chains["small"], chains["generation"], chains["check"]
    models = Models(
        {
            "router": small,
            "rewrite": small,
            "grade": small,
            "react": small,
            "answer": generation,
            "report": generation,
            "ground": check,
        }
    )
    tools = Tools(units, retriever, risk_llm_factory=small)
    return AgentRunner(RunContext(tools, models), sqlite_checkpointer(checkpoints))
