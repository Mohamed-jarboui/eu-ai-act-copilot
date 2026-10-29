"""The citation parser and refusal logic the app uses, gathered in one place."""

from __future__ import annotations

from copilot.answer import citations, refusal

IMPL = "mine"
RETRIEVE_IMPL = "mine"  # the retrieval functions the app uses (rrf_fuse in retrieve/fusion.py)


class Impl:
    def __init__(self, name: str, parse_citation, should_refuse) -> None:
        self.name = name
        self.parse_citation = parse_citation
        self.should_refuse = should_refuse


def get_impl(name: str = IMPL) -> Impl:
    if name != "mine":
        raise ValueError(f"unknown implementation {name!r}")
    return Impl(name, citations.parse_citation, refusal.should_refuse)
