# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for compression axis 3 (eviction regime).

Everything here runs on the CPU: the keep decision is pure bookkeeping over
score tensors, so it needs neither a GPU nor a model. The scores are injected
through ``receive_score`` exactly as a real scorer would deliver them.
"""
import numpy as np
import pytest
import torch

from vllm.v1.attention.compression.compressor import KVCompressor
from vllm.v1.attention.compression.eviction_regime import (
    BudgetRegime,
    ChunkParams,
    RatioRegime,
)
from vllm.v1.attention.compression.workspace import (
    CompressionWorkspace,
    WorkspaceSpec,
)


@pytest.fixture(scope="module", autouse=True)
def single_rank_parallel_state():
    """The selection levels query the tensor-parallel world size to decide
    whether to all-gather. Stand up a one-process gloo group so they can run on
    the CPU; the gather branch is never taken at world size 1."""
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.utils.network_utils import get_open_port

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, backend="gloo",
        distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}")
    ensure_model_parallel_initialized(1, 1)
    yield


NUM_LAYERS = 2
NUM_KV_HEADS = 4
PAGE_GROUP_SIZE = 2
NUM_GROUPS = NUM_KV_HEADS // PAGE_GROUP_SIZE
BLOCK_SIZE = 16
HEAD_SIZE = 8
HIDDEN_DIM = 32


def make_compressor(
    regime: str,
    level: str = "uniform",
    *,
    chunk_size: int = 32,
    budget: int | None = 96,
    window_size: int = 8,
    n_sink_tokens: int = 4,
    evict_current_chunk: bool = False,
    max_num_reqs: int = 2,
) -> KVCompressor:
    spec = WorkspaceSpec.from_config(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        num_groups=NUM_GROUPS,
        page_group_size=PAGE_GROUP_SIZE,
        max_num_reqs=max_num_reqs,
        max_model_len=1 << 20,
        model_dtype=torch.float32,
        chunk_size=chunk_size,
        window_size=window_size,
        n_sink_tokens=n_sink_tokens,
        budget_tokens=budget if regime == "budget" else None,
        evict_current_chunk=evict_current_chunk,
        scorer="snapkv",
    )
    workspace = CompressionWorkspace(spec, torch.device("cpu"))
    compressor = KVCompressor(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        page_group_size=PAGE_GROUP_SIZE,
        head_size=HEAD_SIZE,
        hidden_dim=HIDDEN_DIM,
        block_size=BLOCK_SIZE,
        dtype=torch.float32,
        device="cpu",
        workspace=workspace,
        level=level,
        regime=regime,
    )
    compressor.set_cluster_map(None)
    return compressor


def feed_scores(
    compressor: KVCompressor,
    req_id: str,
    chunk_len: int,
    generator: torch.Generator,
) -> None:
    """Deliver one chunk of per-(head, token) scores for every layer."""
    for layer_idx in range(NUM_LAYERS):
        compressor.receive_score(
            req_id, layer_idx,
            torch.rand(NUM_KV_HEADS, chunk_len, generator=generator))


def run_chunk(
    compressor: KVCompressor,
    req_id: str,
    prev_lens: np.ndarray,
    chunk_len: int,
    params: ChunkParams,
    generator: torch.Generator,
    floor_min: int = 0,
) -> np.ndarray:
    """Score + decide one chunk, returning the post-evict kept lengths.

    Mirrors what ``_run_compression_layer_loop`` does per boundary step, minus
    the KV writeback (which needs a block table and a real cache).
    """
    feed_scores(compressor, req_id, chunk_len, generator)
    compressor.prepare_keep_decision(
        req_id=req_id,
        prev_seq_lens_per_layer=torch.from_numpy(prev_lens),
        chunk_len=chunk_len,
        params=params,
    )
    kept = compressor.compute_kept_lengths_per_rank(
        req_id=req_id,
        eff_seq_lens_row=prev_lens.reshape(-1),
        chunk_len=chunk_len,
        floor_min=floor_min,
    )
    # The executor normally publishes this; without it the once-only assert
    # would reject the next chunk. ``new_locked`` mirrors what the writeback
    # derives: the kept length minus the unconditionally kept regions.
    decision = compressor.req_state[req_id].cross_layer_decision
    new_locked = np.maximum(
        kept.astype(np.int64) - decision.sink_size - decision.tail_size, 0)
    compressor.commit_chunk(req_id, new_locked, kept)
    return kept


def budget_params(
    budget: int,
    *,
    window_size: int = 8,
    n_sink_tokens: int = 4,
    evict_current_chunk: bool = False,
    total_prompt_tokens: int = 0,
) -> ChunkParams:
    return ChunkParams(
        ratio=1.0,
        budget_tokens=budget,
        window_size=window_size,
        n_sink_tokens=n_sink_tokens,
        evict_current_chunk=evict_current_chunk,
        total_prompt_tokens=total_prompt_tokens,
    )


def test_budget_no_eviction_until_budget_is_exceeded():
    """The worked example from the request: budget 64, chunk 32 — the first two
    chunks fit and must be kept whole; the third must come back under budget."""
    budget, chunk_len = 64, 32
    compressor = make_compressor("budget", budget=budget, chunk_size=chunk_len)
    generator = torch.Generator().manual_seed(0)
    compressor.begin_request("r0")

    prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    kept1 = run_chunk(compressor, "r0", prev, chunk_len,
                      budget_params(budget), generator)
    assert np.all(kept1 == chunk_len), "under budget: nothing may be evicted"

    kept2 = run_chunk(compressor, "r0", kept1.astype(np.int64), chunk_len,
                      budget_params(budget), generator)
    assert np.all(kept2 == 2 * chunk_len), "exactly at budget: still no evict"

    kept3 = run_chunk(compressor, "r0", kept2.astype(np.int64), chunk_len,
                      budget_params(budget), generator)
    assert np.all(kept3 <= budget), (
        f"over budget must be cut back to it, got {kept3}")
    assert np.all(kept3 > 2 * chunk_len - chunk_len), (
        "the cut must keep more than just the fresh chunk")


def test_budget_holds_across_many_chunks():
    """A long prompt never drifts above the budget, chunk after chunk."""
    budget, chunk_len = 96, 32
    compressor = make_compressor("budget")
    generator = torch.Generator().manual_seed(1)
    compressor.begin_request("r0")

    prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    for _ in range(10):
        prev = run_chunk(compressor, "r0", prev.astype(np.int64), chunk_len,
                         budget_params(budget), generator).astype(np.int64)
        assert np.all(prev <= budget), f"budget exceeded: {prev}"


def test_budget_protects_the_current_chunk_by_default():
    """With the default protection the fresh chunk survives whole, so the kept
    length is at least sink + chunk; enabling eviction of the current chunk
    shrinks the protected tail to the recent window."""
    budget, chunk_len, sink, window = 96, 32, 4, 8
    generator = torch.Generator().manual_seed(2)

    protected = make_compressor("budget")
    protected.begin_request("r0")
    prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    for _ in range(4):
        prev = run_chunk(
            protected, "r0", prev.astype(np.int64), chunk_len,
            budget_params(budget, window_size=window, n_sink_tokens=sink),
            generator).astype(np.int64)
    assert np.all(prev >= sink + chunk_len)

    evictable = make_compressor("budget", evict_current_chunk=True)
    evictable.begin_request("r0")
    prev_e = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    for _ in range(4):
        prev_e = run_chunk(
            evictable, "r0", prev_e.astype(np.int64), chunk_len,
            budget_params(budget, window_size=window, n_sink_tokens=sink,
                          evict_current_chunk=True),
            generator).astype(np.int64)
    assert np.all(prev_e <= budget)
    # A smaller protected tail leaves more of the budget for scored positions,
    # so the two configurations must not coincide by accident.
    assert protected.req_state["r0"].cross_layer_decision.tail_size == chunk_len
    assert evictable.req_state["r0"].cross_layer_decision.tail_size == window


def test_budget_regime_has_no_lock_in():
    """Under a budget nothing is locked in — every chunk re-ranks the whole
    cache, so the locked count stays zero (which is what makes shrinking back
    to the budget possible at all)."""
    compressor = make_compressor("budget")
    generator = torch.Generator().manual_seed(3)
    compressor.begin_request("r0")
    prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    for _ in range(4):
        prev = run_chunk(compressor, "r0", prev.astype(np.int64), 32,
                         budget_params(96), generator).astype(np.int64)
        assert np.all(compressor.req_state["r0"].locked_count_cpu == 0)


def test_ratio_regime_keeps_lock_in_and_grows():
    """The ratio regime is unchanged: the locked prefix only grows, and so does
    the cache (it converges on ratio * prompt rather than on a cap)."""
    compressor = make_compressor("ratio")
    generator = torch.Generator().manual_seed(4)
    compressor.begin_request("r0")
    params = ChunkParams(
        ratio=0.5, budget_tokens=None, window_size=8, n_sink_tokens=4,
        evict_current_chunk=False, total_prompt_tokens=128)

    prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    locked_prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    lengths = []
    for _ in range(4):
        kept = run_chunk(compressor, "r0", prev.astype(np.int64), 32,
                         params, generator)
        locked_now = compressor.req_state["r0"].locked_count_cpu
        assert np.all(locked_now >= locked_prev), "lock-in must never shrink"
        locked_prev = locked_now
        lengths.append(kept.copy())
        prev = kept.astype(np.int64)
    assert np.all(lengths[-1] >= lengths[0])


def test_budget_regime_rejects_missing_budget():
    """The budget regime is only reachable with a budget set; a missing one is a
    wiring bug, not a fallback to some other target."""
    with pytest.raises(AssertionError):
        BudgetRegime().plan(
            store=None,
            prev_lens=np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64),
            chunk_len=8,
            prev_locked=torch.zeros(NUM_LAYERS, NUM_GROUPS, dtype=torch.long),
            is_first_chunk=True,
            params=ChunkParams(
                ratio=1.0, budget_tokens=None, window_size=8,
                n_sink_tokens=4, evict_current_chunk=False,
                total_prompt_tokens=0),
            device=torch.device("cpu"),
        )


def test_ratio_regime_adjusted_ratio_matches_baseline_formula():
    """The ratio regime's window correction is the reference's
    ``(ratio * clen - window) / (clen - window)`` on the sink-excluded prompt."""
    params = ChunkParams(
        ratio=0.3, budget_tokens=None, window_size=32, n_sink_tokens=4,
        evict_current_chunk=False, total_prompt_tokens=1024)
    got = RatioRegime._adjusted_ratio(params, sink_size=4, win_size=32)
    clen = 1024 - 4
    assert got == pytest.approx((0.3 * clen - 32) / (clen - 32))


def test_persistent_stat_buffer_follows_the_kv():
    """A surviving position's statistics move to its new slot; an evicted
    position's are released. This is what lets a later chunk rank positions
    written by an earlier one."""
    compressor = make_compressor("budget")
    generator = torch.Generator().manual_seed(5)
    compressor.begin_request("r0")

    chunk_len = 32
    prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    run_chunk(compressor, "r0", prev, chunk_len,
              budget_params(96), generator)
    store = compressor.req_state["r0"].score_store
    buffer = store.buffer
    assert buffer is not None
    assert buffer.shape[:2] == (NUM_LAYERS, NUM_KV_HEADS)

    # Keep only the even slots of cluster 0, in order.
    kept_length = 8
    keep_positions = (torch.arange(kept_length, dtype=torch.long) * 2
                      ).unsqueeze(0).expand(PAGE_GROUP_SIZE, -1).contiguous()
    before = buffer.view(NUM_LAYERS * NUM_KV_HEADS, -1).clone()
    compressor.compact_cluster_stats(
        "r0", 0, 0, keep_positions, kept_length)
    after = buffer.view(NUM_LAYERS * NUM_KV_HEADS, -1)

    members = compressor.cluster_members[0].tolist()
    for col, member in enumerate(members):
        torch.testing.assert_close(
            after[member, :kept_length],
            before[member].gather(0, keep_positions[col]))
        assert torch.all(
            after[member, kept_length:] == torch.finfo(buffer.dtype).min), (
            "evicted slots must be released, not left stale")
    # Members of other clusters are untouched.
    untouched = [m for m in range(NUM_LAYERS * NUM_KV_HEADS)
                 if m not in members]
    for member in untouched:
        torch.testing.assert_close(after[member], before[member])


def test_ratio_regime_store_needs_no_compaction():
    """The ratio regime's score memory is chunk-local, so the executor's
    compaction hook must be a safe no-op for it."""
    compressor = make_compressor("ratio")
    generator = torch.Generator().manual_seed(6)
    compressor.begin_request("r0")
    params = ChunkParams(
        ratio=0.5, budget_tokens=None, window_size=8, n_sink_tokens=4,
        evict_current_chunk=False, total_prompt_tokens=128)
    run_chunk(compressor, "r0", np.zeros((NUM_LAYERS, NUM_GROUPS),
                                         dtype=np.int64), 32, params,
              generator)
    compressor.compact_cluster_stats(
        "r0", 0, 0, torch.zeros(PAGE_GROUP_SIZE, 4, dtype=torch.long), 4)


@pytest.mark.parametrize(
    "level", ["uniform", "perlayer_cluster", "crosslayer_cluster",
              "perlayer_head", "crosslayer_head"])
def test_budget_is_respected_for_every_selection_level(level):
    """The budget is enforced whatever scope the level shares it over. The
    cluster-calibrated levels land on it directly; the head-calibrated ones
    would overshoot through their max-pool, so the hard cap must catch them."""
    budget, chunk_len = 96, 32
    compressor = make_compressor("budget", level=level)
    generator = torch.Generator().manual_seed(7)
    compressor.begin_request("r0")
    prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    for _ in range(6):
        prev = run_chunk(compressor, "r0", prev.astype(np.int64), chunk_len,
                         budget_params(budget), generator).astype(np.int64)
        assert np.all(prev <= budget), f"{level} exceeded budget: {prev}"


def test_budget_floor_min_cannot_exceed_the_budget():
    """``compression_floor_min`` is a safety net against a group collapsing, not
    a licence to overshoot the budget."""
    budget, chunk_len = 96, 32
    compressor = make_compressor("budget")
    generator = torch.Generator().manual_seed(8)
    compressor.begin_request("r0")
    prev = np.zeros((NUM_LAYERS, NUM_GROUPS), dtype=np.int64)
    for _ in range(5):
        prev = run_chunk(
            compressor, "r0", prev.astype(np.int64), chunk_len,
            budget_params(budget), generator,
            floor_min=4096).astype(np.int64)
        assert np.all(prev <= budget), f"floor_min broke the budget: {prev}"
