# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the slot score sources (budget-regime score provenance).

CPU only: reading cached keys and scoring them is layout arithmetic plus a
cosine, so a synthetic cache written through the documented column-major layout
is enough — and pins that layout against the helper both the writeback and the
rescoring path use.
"""
import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.v1.attention.compression.keydiff import KeyDiffScorer
from vllm.v1.attention.compression.qk_scorer_base import (
    RESCORE_KEYS,
    RESCORE_VALUES,
    CachedPositions,
)
from vllm.v1.attention.compression.slot_scores import (
    SLOT_SCORE_SOURCE_CHOICES,
    ChunkScoreInputs,
    KVCacheView,
    PersistedChunkScores,
    RecomputedCacheScores,
    SlotFillTarget,
    make_slot_score_source,
)
from vllm.v1.attention.compression.snapkv import SnapKVScorer

NUM_LAYERS = 2
NUM_KV_HEADS = 4
PAGE_GROUP_SIZE = 2
NUM_GROUPS = NUM_KV_HEADS // PAGE_GROUP_SIZE
BLOCK_SIZE = 4
HEAD_SIZE = 8
NUM_BLOCKS = 32


def build_cache_and_view(
    live_lens: np.ndarray,
    generator: torch.Generator,
) -> tuple[KVCacheView, dict[tuple[int, int], torch.Tensor]]:
    """Lay out random keys for every (layer, group) and return a view plus the
    keys that were written, addressed by (layer, group).

    Writes through the documented column-major layout ``[2, num_blocks,
    page_group_size, block_size, head_size]``: slot ``t`` of column ``c`` lives
    at block ``t // block_size``, offset ``t % block_size``.
    """
    caches = [
        torch.zeros(2, NUM_BLOCKS, PAGE_GROUP_SIZE, BLOCK_SIZE, HEAD_SIZE)
        for _ in range(NUM_LAYERS)
    ]
    max_blocks = NUM_BLOCKS
    block_table = torch.zeros(
        1, NUM_LAYERS * NUM_GROUPS, max_blocks, dtype=torch.int32)
    written: dict[tuple[int, int], torch.Tensor] = {}
    next_block = 0
    for layer_idx in range(NUM_LAYERS):
        for group_idx in range(NUM_GROUPS):
            num_positions = int(live_lens[layer_idx, group_idx])
            num_blocks = (num_positions + BLOCK_SIZE - 1) // BLOCK_SIZE
            block_ids = list(range(next_block, next_block + num_blocks))
            next_block += num_blocks
            row = layer_idx * NUM_GROUPS + group_idx
            for slot, block_id in enumerate(block_ids):
                block_table[0, row, slot] = block_id
            keys = torch.rand(
                PAGE_GROUP_SIZE, num_positions, HEAD_SIZE,
                generator=generator)
            for col in range(PAGE_GROUP_SIZE):
                for pos in range(num_positions):
                    caches[layer_idx][
                        0, block_ids[pos // BLOCK_SIZE], col,
                        pos % BLOCK_SIZE] = keys[col, pos]
            written[(layer_idx, group_idx)] = keys
    view = KVCacheView(
        layer_kv_caches=caches,
        block_table_gpu=block_table,
        row_idx=0,
        compressed_layer_ids=np.arange(NUM_LAYERS),
        num_groups=NUM_GROUPS,
        block_size=BLOCK_SIZE,
    )
    return view, written


def test_cache_view_reads_the_documented_layout():
    """``materialize`` must return exactly the keys that were written, in slot
    order — the whole rescoring path rests on this addressing."""
    generator = torch.Generator().manual_seed(0)
    live_lens = np.array([[7, 4], [12, 1]], dtype=np.int64)
    view, written = build_cache_and_view(live_lens, generator)
    for layer_idx in range(NUM_LAYERS):
        for group_idx in range(NUM_GROUPS):
            num_positions = int(live_lens[layer_idx, group_idx])
            got = view.materialize(
                layer_idx, group_idx, num_positions,
                (RESCORE_KEYS, )).require(RESCORE_KEYS)
            assert got.shape == (
                PAGE_GROUP_SIZE, num_positions, HEAD_SIZE)
            torch.testing.assert_close(got, written[(layer_idx, group_idx)])


def test_keydiff_cached_score_is_the_paper_formula():
    """``score_cached`` is the paper's rule: ONE anchor over all the keys handed
    to it, and every key scored against it. The default anchor is the paper's
    experimental setting, the mean of the raw keys."""
    generator = torch.Generator().manual_seed(1)
    keys = torch.rand(PAGE_GROUP_SIZE, 13, HEAD_SIZE, generator=generator)
    scorer = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)

    got = scorer.score_cached(CachedPositions(keys=keys, num_positions=keys.shape[1]))
    anchor = keys.float().mean(dim=1, keepdim=True)
    expected = -F.cosine_similarity(keys.float(), anchor, dim=-1)
    torch.testing.assert_close(got, expected)
    assert got.shape == (PAGE_GROUP_SIZE, 13)


def test_keydiff_whole_cache_anchor_differs_from_the_chunk_anchor():
    """The change the budget regime needs is real: scoring against the whole
    cache's anchor ranks positions differently from scoring each chunk against
    its own, which is why stored per-chunk scores are not interchangeable."""
    generator = torch.Generator().manual_seed(2)
    scorer = KeyDiffScorer(num_kv_heads=1, head_size=HEAD_SIZE)
    # Two chunks with clearly different key distributions.
    chunk_a = torch.rand(1, 16, HEAD_SIZE, generator=generator)
    chunk_b = torch.rand(1, 16, HEAD_SIZE, generator=generator) + 3.0
    whole = torch.cat([chunk_a, chunk_b], dim=1)

    def score(keys: torch.Tensor) -> torch.Tensor:
        return scorer.score_cached(
            CachedPositions(keys=keys, num_positions=keys.shape[1]))

    per_chunk = torch.cat([score(chunk_a), score(chunk_b)], dim=1)
    whole_cache = score(whole)
    per_chunk_rank = per_chunk.argsort(dim=-1)
    whole_rank = whole_cache.argsort(dim=-1)
    assert not torch.equal(per_chunk_rank, whole_rank), (
        "if these agreed, recomputing over the cache would be pointless")


def identity_cluster_maps() -> tuple[torch.Tensor, np.ndarray]:
    """Member row ``layer * num_kv_heads + head`` belongs to cluster
    ``layer * num_groups + head // page_group_size`` at column
    ``head % page_group_size``."""
    member_to_cluster = torch.tensor([
        layer * NUM_GROUPS + head // PAGE_GROUP_SIZE
        for layer in range(NUM_LAYERS) for head in range(NUM_KV_HEADS)])
    cluster_members = np.array([
        [layer * NUM_KV_HEADS + group * PAGE_GROUP_SIZE + col
         for col in range(PAGE_GROUP_SIZE)]
        for layer in range(NUM_LAYERS) for group in range(NUM_GROUPS)])
    return member_to_cluster, cluster_members


def test_recompute_source_fills_every_live_slot():
    """The source must leave a score at every live slot and nothing selectable
    beyond it, for each (layer, group) independently."""
    generator = torch.Generator().manual_seed(3)
    chunk_len = 4
    prev_lens = np.array([[3, 0], [8, 1]], dtype=np.int64)
    live_lens = prev_lens + chunk_len
    view, written = build_cache_and_view(live_lens, generator)

    capacity = 32
    buffer = torch.full(
        (NUM_LAYERS, NUM_KV_HEADS, capacity), float("-inf"))
    flat = buffer.view(NUM_LAYERS * NUM_KV_HEADS, capacity)
    member_to_cluster, cluster_members = identity_cluster_maps()

    scorer = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    source = RecomputedCacheScores(scorer)
    source.fill(
        SlotFillTarget(
            buffer=buffer, flat=flat, member_to_cluster=member_to_cluster,
            cluster_members_cpu=cluster_members, num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS, num_groups=NUM_GROUPS,
            neg_inf=float("-inf")),
        ChunkScoreInputs(
            pending=torch.empty(NUM_LAYERS, NUM_KV_HEADS, 0),
            prev_lens_cpu=prev_lens,
            prev_lens_device=torch.from_numpy(prev_lens),
            chunk_len=chunk_len,
            cache_view=view),
    )

    for layer_idx in range(NUM_LAYERS):
        for group_idx in range(NUM_GROUPS):
            num_positions = int(live_lens[layer_idx, group_idx])
            group_keys = written[(layer_idx, group_idx)]
            expected = scorer.score_cached(CachedPositions(
                keys=group_keys, num_positions=group_keys.shape[1]))
            for col in range(PAGE_GROUP_SIZE):
                head = group_idx * PAGE_GROUP_SIZE + col
                torch.testing.assert_close(
                    buffer[layer_idx, head, :num_positions], expected[col])
                assert torch.all(torch.isinf(
                    buffer[layer_idx, head, num_positions:])), (
                    "slots past the live extent must stay unselectable")


def test_recompute_source_requires_cache_access():
    source = RecomputedCacheScores(
        KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE))
    target = SlotFillTarget(
        buffer=torch.zeros(1, 1, 1), flat=torch.zeros(1, 1),
        member_to_cluster=torch.zeros(1, dtype=torch.long),
        cluster_members_cpu=np.zeros((1, 1), dtype=np.int64),
        num_layers=1, num_kv_heads=1, num_groups=1, neg_inf=0.0)
    with pytest.raises(RuntimeError, match="KVCacheView"):
        source.fill(target, ChunkScoreInputs(
            pending=torch.zeros(1, 1, 0),
            prev_lens_cpu=np.zeros((1, 1), dtype=np.int64),
            prev_lens_device=torch.zeros(1, 1, dtype=torch.long),
            chunk_len=0, cache_view=None))


def test_source_selection_follows_the_scorer():
    """A scorer that can rescore the cache gets the recompute source; one that
    cannot keeps its chunk scores. Not a user-facing choice."""
    keydiff = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    snapkv = SnapKVScorer(
        num_kv_heads=NUM_KV_HEADS, num_q_per_kv=2, head_size=HEAD_SIZE,
        window=8, kernel=3)
    assert isinstance(make_slot_score_source(keydiff), RecomputedCacheScores)
    assert isinstance(make_slot_score_source(snapkv), PersistedChunkScores)
    assert isinstance(make_slot_score_source(None), PersistedChunkScores)
    # A recomputing source makes the per-chunk scorer unnecessary; a persisting
    # one depends on it.
    assert not RecomputedCacheScores(keydiff).needs_chunk_scores
    assert PersistedChunkScores().needs_chunk_scores


def test_budget_regime_skips_chunk_scoring_only_when_recomputing():
    from vllm.v1.attention.compression.eviction_regime import (
        BudgetRegime,
        RatioRegime,
    )
    keydiff = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    recompute = RecomputedCacheScores(keydiff)
    persist = PersistedChunkScores()
    assert not BudgetRegime().consumes_chunk_scores(recompute)
    assert BudgetRegime().consumes_chunk_scores(persist)
    # The ratio regime has only the chunk's own scores to rank, whatever the
    # scorer could do.
    assert RatioRegime().consumes_chunk_scores(recompute)
    assert not RatioRegime().uses_slot_scores
    assert BudgetRegime().uses_slot_scores


def test_forcing_a_source_overrides_the_scorer_for_ablations():
    """A source name pins the provenance so a budget run can rank the same
    scores a ratio run would, isolating the retention target from the score."""
    keydiff = KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE)
    snapkv = SnapKVScorer(
        num_kv_heads=NUM_KV_HEADS, num_q_per_kv=2, head_size=HEAD_SIZE,
        window=8, kernel=3)

    forced = make_slot_score_source(keydiff, "persist")
    assert isinstance(forced, PersistedChunkScores)
    # The chunk scorer must run again: the ablation ranks the scores the chunks
    # produced, so skipping them would leave the buffer empty.
    assert forced.needs_chunk_scores
    # The startup line must not claim the scorer is incapable — it is not.
    assert forced.forced_over_recompute
    assert "forced for an ablation" in forced.describe()

    # Forcing what auto would have picked anyway is a no-op, not an ablation.
    assert isinstance(
        make_slot_score_source(keydiff, "recompute"), RecomputedCacheScores)
    plain = make_slot_score_source(snapkv, "persist")
    assert isinstance(plain, PersistedChunkScores)
    assert not plain.forced_over_recompute


def test_forcing_recompute_on_a_non_rescoring_scorer_is_rejected():
    """Not degradable: the score simply cannot be reconstructed, so failing
    loudly beats ranking cached positions by something else."""
    snapkv = SnapKVScorer(
        num_kv_heads=NUM_KV_HEADS, num_q_per_kv=2, head_size=HEAD_SIZE,
        window=8, kernel=3)
    with pytest.raises(ValueError, match="rescores_cache"):
        make_slot_score_source(snapkv, "recompute")
    with pytest.raises(ValueError, match="rescores_cache"):
        make_slot_score_source(None, "recompute")
    with pytest.raises(ValueError, match="must be one of"):
        make_slot_score_source(snapkv, "accumulate")


def test_config_rejects_a_forced_source_that_would_do_nothing():
    """The setting is a measurement instrument: a run that silently ignores it
    would look like the ablation without being it."""
    from vllm.config.cache import CacheConfig

    def build(**overrides):
        kwargs = dict(page_group_size=PAGE_GROUP_SIZE,
                      compression_budget_tokens=4096,
                      compression_chunk_size=1024,
                      compression_scorer="keydiff")
        kwargs.update(overrides)
        return CacheConfig(**kwargs)

    assert build().compression_slot_score_source == "auto"
    assert "auto" in SLOT_SCORE_SOURCE_CHOICES
    for source in SLOT_SCORE_SOURCE_CHOICES:
        assert build(compression_slot_score_source=source)

    # Ratio regime: nothing ever scores a cached position.
    with pytest.raises(ValueError, match="requires compression_budget_tokens"):
        build(compression_slot_score_source="persist",
              compression_budget_tokens=None, compression_ratio=0.5)
    # Recompute against a scorer that cannot.
    with pytest.raises(ValueError, match="needs a scorer"):
        build(compression_slot_score_source="recompute",
              compression_scorer="snapkv")
    with pytest.raises(ValueError, match="must be one of"):
        build(compression_slot_score_source="accumulate")


def test_scorer_options_are_declared_by_the_scorer():
    """A setting's default, accepted values and type live with the scorer, so
    adding one is a subclass change rather than a config/CLI/factory change."""
    from vllm.v1.attention.compression.scorer import (
        QK_SCORERS,
        build_qk_scorer,
        get_scorer_options,
    )
    from vllm.v1.attention.compression.scorer_options import (
        parse_scorer_options,
        resolve_scorer_options,
    )

    # Every registered scorer builds through the one shared contract.
    for name in QK_SCORERS:
        scorer = build_qk_scorer(
            name, num_kv_heads=NUM_KV_HEADS, num_q_per_kv=2,
            head_size=HEAD_SIZE, options=None)
        assert scorer.name == name

    assert parse_scorer_options("anchor=normalized, window=8") == {
        "anchor": "normalized", "window": "8"}
    assert parse_scorer_options("") == {}
    with pytest.raises(ValueError, match="key=value"):
        parse_scorer_options("anchor")

    # Types come from the declaration, not from the caller.
    resolved = resolve_scorer_options(
        "expected_attention", get_scorer_options("expected_attention"),
        {"use_vnorm": "false", "n_future_positions": "128"})
    assert resolved["use_vnorm"] is False
    assert resolved["n_future_positions"] == 128
    assert resolved["use_covariance"] is True          # untouched default

    with pytest.raises(ValueError, match="unknown scorer option"):
        resolve_scorer_options("keydiff", get_scorer_options("keydiff"),
                               {"anchr": "normalized"})
    with pytest.raises(ValueError, match="expected one of"):
        resolve_scorer_options("keydiff", get_scorer_options("keydiff"),
                               {"anchor": "mean"})


def test_legacy_scorer_flags_agree_with_the_declared_defaults():
    """The pre-existing per-scorer config fields are aliases into the option
    channel, so their defaults must equal what the scorer declares — otherwise
    removing the aliases later would silently change behaviour."""
    from vllm.config.cache import CacheConfig
    from vllm.v1.attention.compression.scorer import get_scorer_options

    for scorer, aliases in CacheConfig._LEGACY_SCORER_OPTION_FIELDS.items():
        declared = {opt.name: opt for opt in get_scorer_options(scorer)}
        for option_name, field_name in aliases.items():
            assert option_name in declared, (
                f"{field_name} aliases {scorer}.{option_name}, which the "
                "scorer does not declare")
            assert getattr(CacheConfig, field_name) == \
                declared[option_name].default, (
                    f"{field_name} default disagrees with "
                    f"{scorer}.{option_name}")


def test_keydiff_anchor_option_selects_the_published_formula():
    """Both entry points must use the SAME anchor: a scorer that ranked the
    fresh chunk by one definition and the cache by another would be neither."""
    generator = torch.Generator().manual_seed(7)
    keys = torch.rand(1, 12, HEAD_SIZE, generator=generator) + 0.5
    flat_key = keys[0].reshape(12, 1 * HEAD_SIZE)

    for anchor_name, anchor_fn in (
        ("unnormalized", lambda k: k.mean(dim=1, keepdim=True)),
        ("normalized",
         lambda k: F.normalize(k, p=2, dim=-1).mean(dim=1, keepdim=True)),
    ):
        scorer = KeyDiffScorer(num_kv_heads=1, head_size=HEAD_SIZE,
                               anchor=anchor_name)
        expected = -F.cosine_similarity(
            keys.float(), anchor_fn(keys.float()), dim=-1)
        cached = scorer.score_cached(
            CachedPositions(keys=keys, num_positions=keys.shape[1]))
        torch.testing.assert_close(cached, expected)
        # forward sees the same tokens as one chunk, so it must agree.
        chunk = scorer(query=torch.empty(0), key=flat_key)
        torch.testing.assert_close(chunk, expected)

    # The two formulas really do rank differently (otherwise the option would
    # be decoration): raw-mean lets a long key pull the anchor towards itself.
    unnorm = KeyDiffScorer(num_kv_heads=1, head_size=HEAD_SIZE,
                           anchor="unnormalized")
    norm = KeyDiffScorer(num_kv_heads=1, head_size=HEAD_SIZE,
                         anchor="normalized")
    skewed = keys.clone()
    skewed[0, 0] *= 20.0
    request = CachedPositions(keys=skewed, num_positions=skewed.shape[1])
    assert not torch.equal(
        unnorm.score_cached(request).argsort(dim=-1),
        norm.score_cached(request).argsort(dim=-1))


def test_rescore_inputs_are_materialized_on_demand():
    """The runner builds exactly what the scorer declared — so a scorer needing
    values or positions can be added without changing this contract again."""
    generator = torch.Generator().manual_seed(11)
    live_lens = np.full((NUM_LAYERS, NUM_GROUPS), 6, dtype=np.int64)
    view, _ = build_cache_and_view(live_lens, generator)

    keys_only = view.materialize(0, 0, 6, (RESCORE_KEYS, ))
    assert keys_only.keys is not None and keys_only.values is None
    assert keys_only.num_positions == 6
    with pytest.raises(RuntimeError, match="rescore_inputs"):
        keys_only.require(RESCORE_VALUES)

    both = view.materialize(0, 0, 6, (RESCORE_KEYS, RESCORE_VALUES))
    assert both.values is not None
    assert both.values.shape == both.keys.shape

    with pytest.raises(ValueError, match="not cache inputs"):
        view.materialize(0, 0, 6, ("logits", ))


def test_persist_source_writes_the_chunk_after_each_cluster_length():
    """Each member's chunk scores land just past its own cluster's pre-chunk
    length, leaving earlier chunks' scores untouched."""
    chunk_len = 4
    capacity = 32
    prev_lens = np.array([[3, 0], [8, 1]], dtype=np.int64)
    buffer = torch.full((NUM_LAYERS, NUM_KV_HEADS, capacity), float("-inf"))
    flat = buffer.view(NUM_LAYERS * NUM_KV_HEADS, capacity)
    member_to_cluster, cluster_members = identity_cluster_maps()
    pending = torch.arange(
        NUM_LAYERS * NUM_KV_HEADS * chunk_len, dtype=torch.float32).view(
            NUM_LAYERS, NUM_KV_HEADS, chunk_len)

    PersistedChunkScores().fill(
        SlotFillTarget(
            buffer=buffer, flat=flat, member_to_cluster=member_to_cluster,
            cluster_members_cpu=cluster_members, num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS, num_groups=NUM_GROUPS,
            neg_inf=float("-inf")),
        ChunkScoreInputs(
            pending=pending,
            prev_lens_cpu=prev_lens,
            prev_lens_device=torch.from_numpy(prev_lens),
            chunk_len=chunk_len,
            cache_view=None),
    )

    for layer_idx in range(NUM_LAYERS):
        for head in range(NUM_KV_HEADS):
            start = int(prev_lens[layer_idx, head // PAGE_GROUP_SIZE])
            torch.testing.assert_close(
                buffer[layer_idx, head, start:start + chunk_len],
                pending[layer_idx, head])
            assert torch.all(
                buffer[layer_idx, head, :start] == float("-inf")), (
                "a write must not reach slots earlier chunks own")


def test_constructor_defaults_match_the_declared_options():
    """``OPTIONS`` owns each default, so a constructor that also states one must
    state the same value — the factory path and direct construction otherwise
    build differently configured scorers from the same declaration."""
    from vllm.v1.attention.compression.scorer import _QK_SCORERS

    for name, scorer_cls in _QK_SCORERS.items():
        signature = inspect.signature(scorer_cls.__init__)
        for option in scorer_cls.OPTIONS:
            parameter = signature.parameters.get(option.name)
            if parameter is None or parameter.default is inspect.Parameter.empty:
                continue
            assert parameter.default == option.default, (
                f"{name}.{option.name}: constructor default "
                f"{parameter.default!r} != declared {option.default!r}")


def test_recompute_source_skips_empty_clusters_and_rejects_partial_ones():
    """Same rule as the score store's compaction: a cross-layer cluster map may
    leave a cluster with no member at all, but a half-filled one would leave the
    members that are there ranking on the previous chunk's scores while their
    peers rank on rescored ones."""
    generator = torch.Generator().manual_seed(7)
    chunk_len = 2
    prev_lens = np.array([[3, 0], [8, 1]], dtype=np.int64)
    view, _ = build_cache_and_view(prev_lens + chunk_len, generator)
    capacity = 32
    buffer = torch.full((NUM_LAYERS, NUM_KV_HEADS, capacity), float("-inf"))
    flat = buffer.view(NUM_LAYERS * NUM_KV_HEADS, capacity)
    member_to_cluster, cluster_members = identity_cluster_maps()
    source = RecomputedCacheScores(
        KeyDiffScorer(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE))

    def fill_with(members: np.ndarray) -> None:
        source.fill(
            SlotFillTarget(
                buffer=buffer, flat=flat, member_to_cluster=member_to_cluster,
                cluster_members_cpu=members, num_layers=NUM_LAYERS,
                num_kv_heads=NUM_KV_HEADS, num_groups=NUM_GROUPS,
                neg_inf=float("-inf")),
            ChunkScoreInputs(
                pending=torch.empty(NUM_LAYERS, NUM_KV_HEADS, 0),
                prev_lens_cpu=prev_lens,
                prev_lens_device=torch.from_numpy(prev_lens),
                chunk_len=chunk_len,
                cache_view=view))

    empty = cluster_members.copy()
    empty[0] = -1
    fill_with(empty)
    assert torch.all(torch.isinf(buffer[0, :PAGE_GROUP_SIZE])), (
        "an empty cluster owns no slots, so none may be written")

    partial = cluster_members.copy()
    partial[0, 1:] = -1
    with pytest.raises(RuntimeError, match="some columns but not others"):
        fill_with(partial)
