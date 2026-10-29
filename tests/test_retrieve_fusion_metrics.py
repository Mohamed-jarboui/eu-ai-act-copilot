"""RRF and the retrieval metrics on tiny fixed examples.

The reference implementation must always pass. Your implementation ("mine": fusion.py and
metrics.py) is marked xfail until you write it: remove the marker once it passes.
"""

import pytest

from copilot.retrieve.impl import get_impl
from copilot.retrieve.metrics import is_hit

TODO = pytest.mark.xfail(reason="TODO(me): fusion.py / metrics.py", strict=False)
IMPLS = ["reference", pytest.param("mine", marks=TODO)]


@pytest.fixture(params=IMPLS)
def impl(request):
    return get_impl(request.param)


# --- the matching rule (implemented, not a TODO) ---------------------------------------------


@pytest.mark.parametrize(
    ("retrieved", "gold", "expected"),
    [
        ("aiact:art:5", "aiact:art:5", True),
        ("aiact:art:5:1:f", "aiact:art:5", True),  # below the gold id
        ("aiact:art:50", "aiact:art:5", False),  # same prefix, different article
        ("aiact:anx:III:2", "aiact:anx:III", True),
        ("aiact:art:5", "aiact:art:5:1", False),  # above the gold id is not a hit
        ("gdpr:art:5", "aiact:art:5", False),
    ],
)
def test_is_hit(retrieved, gold, expected):
    assert is_hit(retrieved, gold) is expected


# --- reciprocal rank fusion --------------------------------------------------------------------


def test_rrf_single_ranking_keeps_order_and_scores(impl):
    fused = impl.rrf_fuse([["a", "b", "c"]], k=1)
    assert [d for d, _ in fused] == ["a", "b", "c"]
    assert [round(s, 6) for _, s in fused] == [round(1 / 2, 6), round(1 / 3, 6), round(1 / 4, 6)]


def test_rrf_documents_in_both_lists_beat_documents_in_one(impl):
    fused = impl.rrf_fuse([["x", "y"], ["y"]], k=60)
    assert [d for d, _ in fused] == ["y", "x"]
    assert dict(fused)["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert dict(fused)["x"] == pytest.approx(1 / 61)


def test_rrf_ties_keep_first_seen_order(impl):
    # a and b swap places between the lists, so they tie; c and d tie at rank 3.
    fused = impl.rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert [d for d, _ in fused] == ["a", "b", "c", "d"]
    assert dict(fused)["a"] == pytest.approx(dict(fused)["b"])


def test_rrf_uses_ranks_not_list_length(impl):
    fused = impl.rrf_fuse([["a"], ["b", "a"]])
    assert [d for d, _ in fused] == ["a", "b"]  # a: 1/61 + 1/62, b: 1/61


def test_rrf_empty(impl):
    assert impl.rrf_fuse([]) == []
    assert impl.rrf_fuse([[], []]) == []


# --- recall@k and reciprocal rank ------------------------------------------------------------


RANKED = ["aiact:art:5:1:f", "aiact:art:6", "gdpr:art:9", "aiact:anx:III:2"]
GOLD = ["aiact:art:5", "aiact:anx:III"]


def test_recall_at_k_counts_gold_ids_found(impl):
    assert impl.recall_at_k(RANKED, GOLD, 1) == pytest.approx(0.5)  # art:5 via its point
    assert impl.recall_at_k(RANKED, GOLD, 3) == pytest.approx(0.5)  # anx:III not yet
    assert impl.recall_at_k(RANKED, GOLD, 4) == pytest.approx(1.0)


def test_recall_counts_a_gold_id_once(impl):
    ranked = ["aiact:art:5:1", "aiact:art:5:2", "aiact:art:5:3"]
    assert impl.recall_at_k(ranked, ["aiact:art:5", "aiact:art:6"], 3) == pytest.approx(0.5)


def test_recall_with_nothing_found(impl):
    assert impl.recall_at_k(["gdpr:art:1"], GOLD, 5) == 0.0
    assert impl.recall_at_k([], GOLD, 5) == 0.0


def test_reciprocal_rank(impl):
    assert impl.reciprocal_rank(RANKED, GOLD) == pytest.approx(1.0)  # first item hits
    assert impl.reciprocal_rank(["gdpr:art:9", "x", "aiact:art:6:2"], ["aiact:art:6"]) == (
        pytest.approx(1 / 3)
    )
    assert impl.reciprocal_rank(["gdpr:art:9"], ["aiact:art:6"]) == 0.0
    assert impl.reciprocal_rank([], ["aiact:art:6"]) == 0.0


def test_reciprocal_rank_takes_the_best_of_several_gold_ids(impl):
    ranked = ["gdpr:art:1", "aiact:anx:III:1", "aiact:art:5"]
    assert impl.reciprocal_rank(ranked, ["aiact:art:5", "aiact:anx:III"]) == pytest.approx(1 / 2)
