# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Axis-1 aggregation: eval scores -> one kept COUNT per (layer, head group).

A group stores ONE length for its members (E3), and that length is the MEAN of
what its members would each have kept (E3b) -- the only rule under which
``sum(cap * page_group_size) == sum(member demand) == the budget``, since a
page carries no padding. The tests below pin that equality, the per-scope
threshold span, and the floor that keeps a group from overspending.
"""
import numpy as np
import pytest
import torch

from vllm.v1.attention.compression import budget_scope
from vllm.v1.attention.compression.budget_scope import make_budget_scope


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    monkeypatch.setattr(
        budget_scope, "get_tensor_model_parallel_world_size", lambda: 1)


def scores_for(demands: list[int], eval_len: int) -> torch.Tensor:
    """Scores whose per-head demand is exactly ``demands``.

    Head ``h`` scores its first ``demands[h]`` positions above every other
    head's remaining positions, so any threshold placed at the overall budget
    cuts each head exactly at its own demand.
    """
    scores = torch.full((1, len(demands), eval_len), -1.0)
    for head, demand in enumerate(demands):
        scores[0, head, :demand] = torch.linspace(1.0, 0.5, demand)
    return scores


def ratio_for(demands: list[int], eval_len: int) -> float:
    """A ratio whose threshold lands just below every demanded position.

    The cut keeps the top ``int(cells * ratio)`` cells; asking for one more
    than the demanded total puts it on the first undemanded cell, so each head
    is cut at exactly its own demand and the test reads what it means to.
    """
    return (sum(demands) + 1) / (len(demands) * eval_len)


def counts_for(scores, ratio, mapping, num_layers, num_heads, num_groups,
               scope="layer"):
    return make_budget_scope(scope).compute_counts(
        scores, ratio, torch.tensor(mapping), num_layers, num_heads,
        num_groups)


# The worked example in the design notes: four heads wanting [4, 8, 2, 10]
# slots, paged two per group. Both groupings spend the same 24 slots; only the
# misallocation differs, which is the whole point of clustering.
CANONICAL_DEMAND = [4, 8, 2, 10]
CANONICAL_EVAL_LEN = 12
CANONICAL_RATIO = ratio_for(CANONICAL_DEMAND, CANONICAL_EVAL_LEN)


@pytest.mark.parametrize("mapping,expected_caps,misallocation", [
    ([0, 0, 1, 1], [6, 6], 12),   # adjacent: {h0,h1} {h2,h3}
    ([0, 1, 0, 1], [3, 9], 4),    # clustered: {h0,h2} {h1,h3}
])
def test_cap_is_the_mean_of_member_demands(mapping, expected_caps,
                                           misallocation):
    counts = counts_for(
        scores_for(CANONICAL_DEMAND, CANONICAL_EVAL_LEN), CANONICAL_RATIO,
        mapping, 1, 4, 2)

    assert counts.reshape(-1).tolist() == expected_caps
    per_head = np.array([counts.reshape(-1)[c] for c in mapping])
    assert int(np.abs(np.array(CANONICAL_DEMAND) - per_head).sum()) == \
        misallocation


@pytest.mark.parametrize("mapping", [[0, 0, 1, 1], [0, 1, 0, 1]])
def test_grouping_does_not_change_the_total(mapping):
    counts = counts_for(
        scores_for(CANONICAL_DEMAND, CANONICAL_EVAL_LEN), CANONICAL_RATIO,
        mapping, 1, 4, 2)

    page_group_size = 2
    assert int(counts.sum()) * page_group_size == sum(CANONICAL_DEMAND)


def test_a_group_never_outspends_its_members():
    """A remainder rounds DOWN: a ceiling would spend slots the budget lacks."""
    demands = [5, 6, 0, 0]  # group 0 wants 5.5 slots per member, group 1 none
    eval_len = 8

    counts = counts_for(scores_for(demands, eval_len),
                        ratio_for(demands, eval_len),
                        [0, 0, 1, 1], 1, 4, 2)

    assert counts.reshape(-1).tolist() == [5, 0]
    assert int(counts.sum()) * 2 <= sum(demands)


def test_layer_scope_thresholds_each_layer_on_its_own():
    """Layer 1's larger scores must not take budget away from layer 0."""
    scores = torch.zeros(2, 2, 4)
    scores[0] = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    scores[1] = scores[0] * 100.0

    counts = counts_for(scores, 0.5, [0, 0, 1, 1], 2, 2, 1)

    layer0, layer1 = counts.reshape(-1).tolist()
    assert layer0 == layer1


def test_global_scope_lets_a_strong_layer_keep_more():
    scores = torch.zeros(2, 2, 4)
    scores[0] = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    scores[1] = scores[0] * 100.0

    counts = counts_for(scores, 0.5, [0, 0, 1, 1], 2, 2, 1, scope="global")

    layer0, layer1 = counts.reshape(-1).tolist()
    assert layer1 > layer0


def test_cross_layer_cluster_shares_one_cap():
    """A global map's cluster may span layers; its members still share a cap."""
    scores = torch.zeros(2, 2, 4)
    scores[0] = torch.tensor([[4.0, 3.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    scores[1] = torch.tensor([[4.0, 3.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    # cluster 0 = the strong head of each layer, cluster 1 = the weak pair.
    counts = counts_for(scores, 0.5, [0, 1, 0, 1], 2, 2, 1, scope="global")

    strong, weak = counts.reshape(-1).tolist()
    assert strong > weak


def test_uniform_scope_gives_every_group_the_same_count():
    counts = make_budget_scope("uniform").compute_counts(
        scores_for(CANONICAL_DEMAND, CANONICAL_EVAL_LEN), 0.5,
        torch.tensor([0, 0, 1, 1]), 1, 4, 2)

    assert counts.reshape(-1).tolist() == [6, 6]


@pytest.mark.parametrize("scope", ["layer", "global"])
def test_tensor_parallel_is_rejected(monkeypatch, scope):
    monkeypatch.setattr(
        budget_scope, "get_tensor_model_parallel_world_size", lambda: 2)

    with pytest.raises(NotImplementedError, match="TP=1"):
        counts_for(scores_for(CANONICAL_DEMAND, CANONICAL_EVAL_LEN), 0.5,
                   [0, 0, 1, 1], 1, 4, 2, scope=scope)
