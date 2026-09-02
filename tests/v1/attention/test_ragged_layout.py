# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index-math equivalence tests for the column-major ragged layout.

These run on CPU with no model or CUDA: they prove that the column-major
virtual-block addressing writes and reads exactly the same logical cells as the
column-minor per-cluster path. Here we verify the wiring (slot and block-table
arithmetic) the metadata builder relies on.
"""
import dataclasses
import os

import torch

import numpy as np

from vllm.v1.attention.backends.ragged_layout import (
    RaggedStepViews,
    as_virtual_block_view,
    build_decode_layer_overlays,
    column_major_cache_shape,
    identity_member_maps,
    load_cluster_map,
    member_maps_from_cluster_map,
    member_seq_lens,
    member_virtual_block_table,
    member_virtual_slots,
)
from ragged_reference import (
    expand_member_seq_lens,
    identity_member_clusters,
    identity_member_columns,
    physical_to_virtual_block_table,
    physical_to_virtual_slots,
)

# Repo-bundled cluster map (tools/head_group_clustering/cluster_maps/). Derived
# from __file__ so the test is self-contained: tests/v1/attention/ -> repo root
# is three levels up.
_CLUSTER_MAP_NPZ = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "tools", "head_group_clustering", "cluster_maps",
    "qwen25-7b-1m_r0.3_pg4.npz",
)


def _emulate_reshape_and_cache(cache_view, slot_mapping, new_kv):
    """Mimic ``reshape_and_cache_flash`` scatter semantics on CPU.

    ``cache_view`` is ``[num_blocks, block_size, num_kv_heads, head_size]``;
    ``new_kv`` is ``[num_entries, num_kv_heads, head_size]``; ``slot_mapping``
    is ``[num_entries]`` with ``slot = block * block_size + offset``.
    """
    num_blocks, block_size, num_kv_heads, head_size = cache_view.shape
    flat = cache_view.reshape(num_blocks * block_size, num_kv_heads, head_size)
    flat[slot_mapping] = new_kv


def test_identity_member_mapping():
    # KV head m -> cluster m // g at column m % g.
    g = 4
    num_kv_heads = 12
    cols = identity_member_columns(num_kv_heads, g)
    clusters = identity_member_clusters(num_kv_heads, g)
    assert cols.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    assert clusters.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_virtual_block_view_addresses_column_major_cells():
    # vv[block * g + col, offset] must alias column-major cache[block, col,
    # offset] for every (block, col, offset).
    torch.manual_seed(0)
    num_blocks, g, block_size, head_size = 5, 4, 16, 8
    cache = torch.randn(*column_major_cache_shape(
        num_blocks, block_size, g, head_size))  # [2, nb, g, bs, hs]
    key_cache = cache[0]  # [nb, g, bs, hs]
    vv = as_virtual_block_view(cache)[0]  # [nb * g, bs, 1, hs]
    for block in range(num_blocks):
        for col in range(g):
            vb = block * g + col
            assert torch.equal(
                vv[vb, :, 0, :], key_cache[block, col, :, :]), (
                f"virtual block {vb} != physical (block {block}, col {col})")


def test_per_member_write_matches_column_minor_path():
    """Writing per-member into the column-major virtual view lands the same
    per-(cluster, column, token) data as the current column-minor per-cluster
    write."""
    torch.manual_seed(1)
    num_phys_blocks, g, block_size, head_size = 6, 4, 16, 8
    num_clusters, num_tokens = 3, 20
    num_kv_heads = num_clusters * g

    # Physical write slots for each (cluster, token): pack tokens contiguously
    # into the cluster's own physical blocks. Cluster c uses blocks
    # [c * blocks_per_cluster ...].
    blocks_per_cluster = (num_tokens + block_size - 1) // block_size
    group_slots = torch.empty(num_clusters, num_tokens, dtype=torch.int64)
    for c in range(num_clusters):
        base_block = c * blocks_per_cluster
        for t in range(num_tokens):
            block = base_block + t // block_size
            offset = t % block_size
            group_slots[c, t] = block * block_size + offset

    # Per-(cluster, column, token) key data: distinct per column so a column
    # mix-up would be caught.
    new_kv = torch.randn(num_clusters, g, num_tokens, head_size)

    # --- Reference: column-minor cache, per-cluster write (one slot writes
    # all g heads of the cluster at that block/offset). ---
    cm = torch.zeros(
        2, num_phys_blocks, block_size, g, head_size)  # [2, nb, bs, g, hs]
    cm_key = cm[0]
    # Cluster c writes its [g, hs] vector for token t at (block, offset).
    # new_kv is [cluster, col, token, hs] -> per token a [g, hs] block.
    cm_new = new_kv.permute(0, 2, 1, 3).reshape(
        num_clusters * num_tokens, g, head_size)
    cm_slots = group_slots.reshape(num_clusters * num_tokens)
    _emulate_reshape_and_cache(cm_key, cm_slots, cm_new)

    # --- Column-major cache, per-member virtual write. ---
    cmaj = torch.zeros(*column_major_cache_shape(
        num_phys_blocks, block_size, g, head_size))  # [2, nb, g, bs, hs]
    virtual_slots = physical_to_virtual_slots(
        group_slots, g, block_size)  # [num_kv_heads, T]
    assert tuple(virtual_slots.shape) == (num_kv_heads, num_tokens)
    vv = as_virtual_block_view(cmaj)[0]  # [nb * g, bs, 1, hs]
    # Member-major data: row (cluster * g + col).
    member_kv = new_kv.reshape(num_kv_heads, num_tokens, head_size).unsqueeze(2)
    _emulate_reshape_and_cache(
        vv, virtual_slots.reshape(-1),
        member_kv.reshape(num_kv_heads * num_tokens, 1, head_size))

    # --- Equivalence: read back per (cluster, col, token) from both caches. ---
    for c in range(num_clusters):
        for col in range(g):
            for t in range(num_tokens):
                block = (c * blocks_per_cluster) + t // block_size
                offset = t % block_size
                ref = cm_key[block, offset, col, :]
                got = cmaj[0, block, col, offset, :]
                assert torch.equal(ref, got), (
                    f"mismatch at cluster {c} col {col} token {t}")
                assert torch.equal(ref, new_kv[c, col, t, :])


def test_per_member_block_table_virtual_ids():
    # Virtual block id of member (cluster, col) reading the cluster's physical
    # block p must be p * g + col, member-major rows.
    g = 4
    num_clusters, max_blocks = 3, 5
    phys_bt = torch.arange(
        num_clusters * max_blocks, dtype=torch.int32).reshape(
        num_clusters, max_blocks)
    virtual_bt = physical_to_virtual_block_table(phys_bt, g)
    assert tuple(virtual_bt.shape) == (num_clusters * g, max_blocks)
    # FlashAttention requires int32 block tables; the dtype must be preserved.
    assert virtual_bt.dtype == torch.int32
    for c in range(num_clusters):
        for col in range(g):
            member = c * g + col
            expected = phys_bt[c] * g + col
            assert torch.equal(virtual_bt[member], expected)


def test_compression_gather_scatter_column_major_roundtrip():
    """The compression executor gathers a cluster's blocks into a token-major
    slab and scatters the kept tokens back. Verify the column-major
    permute+reshape (executor.py) yields a token-major slab aliasing the right
    cells and that an identity keep round-trips the cache unchanged."""
    torch.manual_seed(3)
    g, block_size, head_size = 4, 16, 8
    # Block-aligned span so an identity keep needs no tail padding and must
    # round-trip the cache exactly.
    n_blocks, total_seen = 3, 48
    # Column-major cache for one layer: [2, n_blocks, g, block_size, head_size].
    kv_cache = torch.randn(2, n_blocks, g, block_size, head_size)
    block_ids = torch.arange(n_blocks)

    # Gather exactly as executor.py does.
    slab = (
        kv_cache[:, block_ids]
        .permute(0, 1, 3, 2, 4)
        .reshape(2, -1, g, head_size)[:, :total_seen]
    )
    assert tuple(slab.shape) == (2, total_seen, g, head_size)
    for t in range(total_seen):
        block, offset = t // block_size, t % block_size
        for c in range(g):
            assert torch.equal(slab[:, t, c, :], kv_cache[:, block, c, offset, :])

    # Identity keep (first total_seen positions), padded to whole blocks, then
    # scatter back exactly as executor.py does.
    keep_idx = torch.arange(total_seen)
    kept_kv = slab.index_select(1, keep_idx)
    n_blocks_write = (total_seen + block_size - 1) // block_size
    padded = n_blocks_write * block_size
    if total_seen < padded:
        pad = torch.zeros(2, padded - total_seen, g, head_size)
        kept_kv = torch.cat([kept_kv, pad], dim=1)
    before = kv_cache.clone()
    kv_cache[:, block_ids[:n_blocks_write]] = (
        kept_kv.view(2, n_blocks_write, block_size, g, head_size)
        .permute(0, 1, 3, 2, 4)
    )
    # The first total_seen tokens are unchanged; padding zeroed the tail of the
    # last block (positions total_seen .. padded-1).
    assert torch.equal(kv_cache, before), (
        "identity keep with no tail padding must round-trip the cache")


def test_expansion_along_decode_axis():
    # build()'s decode layout expands the per-layer group axis (axis 2 of
    # [num_layers, num_reqs, n_per_layer, ...]). Verify block-table and
    # seq-len expansion on a non-zero axis reproduces the member-major order.
    g = 4
    num_layers, num_reqs, num_groups, max_blocks = 2, 3, 2, 5
    phys_bt = torch.arange(
        num_layers * num_reqs * num_groups * max_blocks,
        dtype=torch.int32).reshape(num_layers, num_reqs, num_groups, max_blocks)
    virtual_bt = physical_to_virtual_block_table(phys_bt, g, group_axis=2)
    assert tuple(virtual_bt.shape) == (
        num_layers, num_reqs, num_groups * g, max_blocks)
    assert virtual_bt.dtype == torch.int32
    for grp in range(num_groups):
        for col in range(g):
            member = grp * g + col
            expected = phys_bt[:, :, grp, :] * g + col
            assert torch.equal(virtual_bt[:, :, member, :], expected)

    seq_lens = torch.arange(
        num_layers * num_reqs * num_groups, dtype=torch.int32).reshape(
        num_layers, num_reqs, num_groups)
    member_seq_lens = expand_member_seq_lens(seq_lens, g, group_axis=2)
    assert tuple(member_seq_lens.shape) == (
        num_layers, num_reqs, num_groups * g)
    for grp in range(num_groups):
        for col in range(g):
            assert torch.equal(
                member_seq_lens[:, :, grp * g + col], seq_lens[:, :, grp])


def test_gather_identity_matches_repeat_interleave_expansion():
    """The cluster-map gather with the identity map must reproduce the
    contiguous repeat-interleave expansion exactly — identity is one instance of
    the general mechanism, so swapping build() to the gather path cannot regress
    the verified adjacent-head behaviour."""
    num_layers, num_kv_heads, g = 3, 4, 2
    num_clusters = num_layers * num_kv_heads // g
    block_size = 16
    mtc, mtcol = identity_member_maps(num_layers, num_kv_heads, g)
    assert mtc.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert mtcol.tolist() == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # Block table [num_clusters, num_reqs, max_blocks].
    phys_bt = torch.arange(
        num_clusters * 3 * 5, dtype=torch.int32).reshape(num_clusters, 3, 5)
    gathered = member_virtual_block_table(phys_bt, mtc, mtcol, g, cluster_axis=0)
    expanded = physical_to_virtual_block_table(phys_bt, g, group_axis=0)
    assert torch.equal(gathered, expanded)
    assert gathered.dtype == torch.int32

    # Slots [num_clusters, num_tokens].
    phys_slots = torch.arange(
        num_clusters * 40, dtype=torch.int64).reshape(num_clusters, 40)
    g_slots = member_virtual_slots(
        phys_slots, mtc, mtcol, g, block_size, cluster_axis=0)
    e_slots = physical_to_virtual_slots(phys_slots, g, block_size, group_axis=0)
    assert torch.equal(g_slots, e_slots)

    # Seq lens [num_clusters, num_reqs].
    seq = torch.arange(num_clusters * 3, dtype=torch.int32).reshape(num_clusters, 3)
    assert torch.equal(
        member_seq_lens(seq, mtc, cluster_axis=0),
        expand_member_seq_lens(seq, g, group_axis=0))


def test_real_cluster_map_bijection_and_roundtrip():
    """Load the profiler's cross-layer cluster map and verify (a) it is a
    bijection over (cluster, column), and (b) writing each head's KV to its
    mapped cluster/column and reading it back round-trips — the read/write
    addressing the runtime relies on for a non-identity map."""
    if not os.path.exists(_CLUSTER_MAP_NPZ):
        print(f"SKIP: cluster map not found at {_CLUSTER_MAP_NPZ}")
        return
    d = np.load(_CLUSTER_MAP_NPZ, allow_pickle=True)
    cluster_of = torch.from_numpy(d["cluster_of"])
    column_of = torch.from_numpy(d["column_of"])
    g = int(d["page_group_size"])
    num_layers, num_kv_heads = cluster_of.shape
    num_clusters = num_layers * num_kv_heads // g

    mtc, mtcol = member_maps_from_cluster_map(cluster_of, column_of)
    # Bijection: every (cluster, column) used exactly once across all members.
    slots = (mtc * g + mtcol).tolist()
    assert len(set(slots)) == num_layers * num_kv_heads
    assert min(slots) == 0 and max(slots) == num_layers * num_kv_heads - 1

    # Round-trip: one block per cluster, distinct data per head, write each head
    # to its mapped (cluster, column) virtual block, read back per head.
    torch.manual_seed(0)
    block_size, head_size = 16, 8
    n_tok = 12
    cmaj = torch.zeros(*column_major_cache_shape(
        num_clusters, block_size, g, head_size))  # [2, nc, g, bs, hs]
    vv = as_virtual_block_view(cmaj)[0]            # [nc * g, bs, 1, hs]
    flat = vv.reshape(num_clusters * g * block_size, 1, head_size)

    # Per-member key data; physical slot = cluster's block 0 at token offset.
    num_members = num_layers * num_kv_heads
    member_kv = torch.randn(num_members, n_tok, head_size)
    # group_slots[cluster, token] = cluster_block(=cluster) * block_size + token.
    group_slots = (
        torch.arange(num_clusters, dtype=torch.int64)[:, None] * block_size
        + torch.arange(n_tok, dtype=torch.int64)[None, :])
    vslots = member_virtual_slots(
        group_slots, mtc, mtcol, g, block_size, cluster_axis=0)  # [members, n_tok]
    flat[vslots.reshape(-1)] = member_kv.reshape(num_members * n_tok, 1, head_size)

    # Read back: member m lives at cluster mtc[m], column mtcol[m].
    for m in range(num_members):
        cluster, col = int(mtc[m]), int(mtcol[m])
        got = cmaj[0, cluster, col, :n_tok, :]
        assert torch.equal(got, member_kv[m]), f"member {m} round-trip mismatch"
    print(f"real map: {num_clusters} clusters, g={g}, "
          f"{num_layers}x{num_kv_heads} heads — bijection + round-trip OK")


def test_load_cluster_map_validates_and_matches_raw():
    """``load_cluster_map`` returns the same (cluster_of, column_of) as a raw
    load and rejects mismatched page_group_size / num_kv_heads."""
    if not os.path.exists(_CLUSTER_MAP_NPZ):
        print(f"SKIP: cluster map not found at {_CLUSTER_MAP_NPZ}")
        return
    d = np.load(_CLUSTER_MAP_NPZ, allow_pickle=True)
    g = int(d["page_group_size"])
    num_layers, num_kv_heads = d["cluster_of"].shape

    cluster_of, column_of = load_cluster_map(_CLUSTER_MAP_NPZ, g, num_kv_heads)
    assert cluster_of.dtype == torch.int64 and column_of.dtype == torch.int64
    assert cluster_of.shape == (num_layers, num_kv_heads)
    assert torch.equal(cluster_of, torch.from_numpy(d["cluster_of"]).to(torch.int64))
    assert torch.equal(column_of, torch.from_numpy(d["column_of"]).to(torch.int64))

    # page_group_size mismatch and num_kv_heads mismatch both raise.
    for bad in ((g + 1, num_kv_heads), (g, num_kv_heads + 1)):
        try:
            load_cluster_map(_CLUSTER_MAP_NPZ, *bad)
            raise AssertionError(f"load_cluster_map accepted bad args {bad}")
        except ValueError:
            pass
    print(f"load_cluster_map: validated + matched raw "
          f"({num_layers}x{num_kv_heads}, g={g})")


# --- Per-layer decode overlays --------------------------------------------


@dataclasses.dataclass
class _StepMetadata:
    """The five per-layer fields plus one shared field, to show the shared one
    survives the copy."""

    num_actual_tokens: int = 0
    block_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    max_seq_len: int = 123


def _decode_views(num_layers: int, num_reqs: int, num_kv_heads: int,
                  page_group_size: int, max_blocks: int) -> RaggedStepViews:
    """Uniform-decode views over an identity cluster map."""
    num_clusters_per_layer = num_kv_heads // page_group_size
    num_clusters_total = num_layers * num_clusters_per_layer
    members = torch.arange(num_layers * num_kv_heads, dtype=torch.int64)
    return RaggedStepViews(
        num_layers_local=num_layers,
        ragged_decode_layout=True,
        # Distinct ids so a mis-indexed layer is visible, not merely wrong.
        cluster_block_table=torch.arange(
            num_reqs * num_clusters_total * max_blocks, dtype=torch.int32
        ).reshape(num_reqs, num_clusters_total, max_blocks),
        clusters_per_layer=(members // page_group_size).reshape(
            num_layers, num_kv_heads),
        cols_per_layer=(members % page_group_size).reshape(
            num_layers, num_kv_heads),
        seq_lens_grouped=torch.arange(
            num_layers * num_reqs * num_kv_heads, dtype=torch.int32
        ).reshape(num_layers, num_reqs, num_kv_heads),
        slot_mapping_grouped=torch.arange(
            num_layers * num_reqs * num_kv_heads, dtype=torch.int64
        ).reshape(num_layers, num_reqs, num_kv_heads),
        query_start_loc_grouped=torch.arange(
            num_reqs * num_kv_heads + 1, dtype=torch.int32),
    )


def test_decode_overlays_give_each_layer_its_own_five_fields():
    """One overlay per layer, each carrying that layer's members flattened to
    ``num_reqs * num_kv_heads`` sequences, with everything else shared."""
    num_layers, num_reqs, num_kv_heads, g, max_blocks = 3, 2, 4, 2, 5
    views = _decode_views(num_layers, num_reqs, num_kv_heads, g, max_blocks)
    metadata = _StepMetadata()

    overlays = build_decode_layer_overlays(
        metadata, views,
        num_reqs=num_reqs,
        num_kv_heads_per_layer=num_kv_heads,
        page_group_size=g,
    )

    assert len(overlays) == num_layers
    num_virtual_seqs = num_reqs * num_kv_heads
    for layer_idx, overlay in enumerate(overlays):
        assert overlay.num_actual_tokens == num_virtual_seqs
        assert overlay.block_table.shape == (num_virtual_seqs, max_blocks)
        assert torch.equal(
            overlay.seq_lens,
            views.seq_lens_grouped[layer_idx].reshape(num_virtual_seqs))
        assert torch.equal(
            overlay.slot_mapping,
            views.slot_mapping_grouped[layer_idx].reshape(-1))
        assert torch.equal(overlay.query_start_loc,
                           views.query_start_loc_grouped)
        # Shared, not per-layer, and the original is untouched.
        assert overlay.max_seq_len == 123
    assert metadata.num_actual_tokens == 0
    print(f"decode overlays: {num_layers} layers x {num_virtual_seqs} seqs OK")


def test_decode_overlay_block_table_matches_the_member_expansion():
    """Each overlay row is the member's own virtual block table, so it must
    equal what ``member_virtual_block_table`` produces for that (layer, head)."""
    num_layers, num_reqs, num_kv_heads, g, max_blocks = 2, 3, 4, 2, 3
    views = _decode_views(num_layers, num_reqs, num_kv_heads, g, max_blocks)

    overlays = build_decode_layer_overlays(
        _StepMetadata(), views,
        num_reqs=num_reqs,
        num_kv_heads_per_layer=num_kv_heads,
        page_group_size=g,
    )

    for layer_idx, overlay in enumerate(overlays):
        expected = member_virtual_block_table(
            views.cluster_block_table,
            views.clusters_per_layer[layer_idx],
            views.cols_per_layer[layer_idx],
            g,
            cluster_axis=1,
        ).reshape(num_reqs * num_kv_heads, -1)
        assert torch.equal(overlay.block_table, expected), layer_idx
    print("decode overlay block tables match the member expansion")


def test_decode_overlays_reject_the_member_major_layout():
    """The member-major layout shapes seq_lens_grouped differently, so indexing
    it by layer would read the wrong rows rather than fail."""
    num_layers, num_reqs, num_kv_heads, g, max_blocks = 2, 2, 4, 2, 3
    views = _decode_views(num_layers, num_reqs, num_kv_heads, g, max_blocks)
    views = dataclasses.replace(views, ragged_decode_layout=False)

    try:
        build_decode_layer_overlays(
            _StepMetadata(), views,
            num_reqs=num_reqs,
            num_kv_heads_per_layer=num_kv_heads,
            page_group_size=g,
        )
        raise AssertionError("accepted the member-major layout")
    except AssertionError as error:
        assert "uniform-decode layout" in str(error), error
    print("decode overlays reject the member-major layout")
