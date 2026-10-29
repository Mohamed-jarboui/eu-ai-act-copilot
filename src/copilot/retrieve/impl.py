"""The retrieval functions the evaluation and the app use, gathered in one place."""

from __future__ import annotations

from types import SimpleNamespace

from copilot.retrieve import fusion, metrics


def get_impl(name: str = "mine") -> SimpleNamespace:
    """rrf_fuse, recall_at_k and reciprocal_rank from fusion.py and metrics.py."""
    if name != "mine":
        raise ValueError(f"unknown implementation {name!r}")
    return SimpleNamespace(
        name=name,
        rrf_fuse=fusion.rrf_fuse,
        recall_at_k=metrics.recall_at_k,
        reciprocal_rank=metrics.reciprocal_rank,
    )
