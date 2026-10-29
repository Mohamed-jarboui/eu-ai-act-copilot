"""Choose which implementation of the TODO(me) functions the evaluation uses."""

from __future__ import annotations

from types import ModuleType

from copilot.retrieve import _reference, fusion, metrics


class Impl:
    def __init__(self, name: str, fuse: ModuleType, scoring: ModuleType) -> None:
        self.name = name
        self.rrf_fuse = fuse.rrf_fuse
        self.recall_at_k = scoring.recall_at_k
        self.reciprocal_rank = scoring.reciprocal_rank


def get_impl(name: str) -> Impl:
    """ "reference" is mine (_reference.py); "mine" is yours (fusion.py and metrics.py)."""
    if name == "reference":
        return Impl(name, _reference, _reference)
    if name == "mine":
        return Impl(name, fusion, metrics)
    raise ValueError(f"unknown implementation {name!r}")
