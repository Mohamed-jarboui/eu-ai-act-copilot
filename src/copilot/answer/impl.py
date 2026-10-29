"""Choose which implementation of the TODO(me) functions the app uses."""

from __future__ import annotations

from copilot.answer import _reference, citations, refusal

# TODO(me): "reference" = _reference.py (mine). Switch to "mine" (citations.py and refusal.py)
# once their tests pass; the app should behave the same.
IMPL = "reference"
# Same idea for the retrieval functions the app uses (rrf_fuse in retrieve/fusion.py).
RETRIEVE_IMPL = "reference"


class Impl:
    def __init__(self, name: str, parse_citation, should_refuse) -> None:
        self.name = name
        self.parse_citation = parse_citation
        self.should_refuse = should_refuse


def get_impl(name: str = IMPL) -> Impl:
    if name == "reference":
        return Impl(name, _reference.parse_citation, _reference.should_refuse)
    if name == "mine":
        return Impl(name, citations.parse_citation, refusal.should_refuse)
    raise ValueError(f"unknown implementation {name!r}")
