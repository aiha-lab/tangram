# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Column-major ragged KV cache layout and virtual-block addressing.

The one place the virtual-block arithmetic lives: allocation, the
FlashAttention metadata builder and the attention forward path all address the
cache through these helpers.

A page is stored column-major, the per-page column dimension OUTSIDE
``block_size`` -- ``[2, num_physical_blocks, page_group_size, block_size,
head_size]`` -- so every ``(physical_block, column)`` pair is a fully
contiguous single-head block addressed by
``virtual_block_id = physical_block * page_group_size + column``. Flattening
the first two axes therefore yields exactly the standard single-KV-head paged
layout, which FlashAttention consumes with no kernel change and full
coalescing.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


def column_major_cache_shape(
    num_blocks: int,
    block_size: int,
    page_group_size: int,
    head_size: int,
) -> tuple[int, int, int, int, int]:
    """Physical shape of a column-major ragged KV cache tensor.

    The leading ``2`` is the key/value split. The page_group_size (column)
    dimension precedes block_size so that flattening ``(num_blocks,
    page_group_size)`` produces the contiguous virtual-block axis.
    """
    return (2, num_blocks, page_group_size, block_size, head_size)


def as_virtual_block_view(kv_cache: torch.Tensor) -> torch.Tensor:
    """View a column-major cache as the standard single-KV-head paged layout.

    ``kv_cache`` is ``[2, num_blocks, page_group_size, block_size, head_size]``
    and contiguous; the result is ``[2, num_blocks * page_group_size,
    block_size, 1, head_size]`` where the second axis is the virtual block id.
    Used to feed ``reshape_and_cache_flash`` and ``flash_attn_varlen_func``.
    """
    two, num_blocks, page_group_size, block_size, head_size = kv_cache.shape
    assert two == 2, (
        f"column-major ragged cache must lead with the key/value axis "
        f"of size 2; got shape {tuple(kv_cache.shape)}.")
    return kv_cache.reshape(
        2, num_blocks * page_group_size, block_size, 1, head_size)


def cluster_pages_token_major(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    key_only: bool = False,
) -> torch.Tensor:
    """Gather one cluster's pages and present them token-major within a block.

    ``kv_cache`` is the column-major cache ``[2, num_blocks, page_group_size,
    block_size, head_size]``. Result
    ``[2, n_blocks, block_size, page_group_size, head_size]``, or without the
    leading axis for ``key_only`` (all a key-based score reads), so token ``t``
    of column ``c`` sits at ``[t // block_size, t % block_size, c]``. The
    permute is free; the gather is the unavoidable strided read of the
    cluster's pages.

    Single source of truth for how compression reads cached KV -- the eviction
    writeback and the cache-rescoring source both come through here, so they
    cannot disagree about the layout.
    """
    pages = kv_cache[0, block_ids] if key_only else kv_cache[:, block_ids]
    # (blocks, columns, tokens, head) -> (blocks, tokens, columns, head)
    return pages.permute(0, 2, 1, 3) if key_only else pages.permute(
        0, 1, 3, 2, 4)


# --- Cluster-map-driven member addressing ----------------------------------
#
# A real cluster map assigns each (layer, head) an arbitrary (cluster, column)
# and may span layers, so these address members through explicit flat index
# tensors -- ``member_to_cluster[m]`` / ``member_to_col[m]`` at member row
# ``m = layer * num_kv_heads + head`` -- with identity as one instance. Only
# placement changes, so uncompressed output is independent of the map.


def _broadcast_along(vector: torch.Tensor, ndim: int, axis: int) -> torch.Tensor:
    """Reshape a 1-D ``vector`` to broadcast along ``axis`` of an ``ndim``
    tensor (size 1 on every other axis)."""
    shape = [1] * ndim
    shape[axis] = vector.shape[0]
    return vector.view(shape)


def identity_member_maps(
    num_layers: int,
    num_kv_heads: int,
    page_group_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identity cluster map as flat ``(member_to_cluster, member_to_col)``,
    both ``[num_layers * num_kv_heads]``. Head ``h`` of layer ``L`` goes to
    cluster ``L * (num_kv_heads // page_group_size) + h // page_group_size`` at
    column ``h % page_group_size``. ``member_to_cluster`` is int64 for
    ``index_select``.
    """
    num_groups = num_kv_heads // page_group_size
    layers = torch.arange(
        num_layers, device=device, dtype=torch.int64
    ).repeat_interleave(num_kv_heads)
    heads = torch.arange(
        num_kv_heads, device=device, dtype=torch.int64
    ).repeat(num_layers)
    member_to_cluster = layers * num_groups + torch.div(
        heads, page_group_size, rounding_mode="floor")
    member_to_col = heads % page_group_size
    return member_to_cluster, member_to_col


def _validate_bijection(
    cluster_flat,
    column_flat,
    *,
    num_clusters: int,
    page_group_size: int,
    subject: str,
) -> None:
    """Reject a member -> (cluster, column) map that is not a bijection onto
    full clusters.

    Two properties, and breaking either silently corrupts KV rather than
    failing: every (cluster, column) slot must be occupied exactly once, or two
    KV heads write the same page column and one overwrites the other; and every
    cluster must be full to ``page_group_size``, or a page carries a column no
    head reads while the budget was still spent on it.

    Both flat arrays are int64 over member rows. ``subject`` names the map in
    the error -- a file's contents and an engine-derived map reach this from
    different places and the reader needs to know which.
    """
    import numpy as np

    if cluster_flat.min() < 0 or cluster_flat.max() >= num_clusters:
        raise ValueError(
            f"{subject} cluster ids out of range [0, {num_clusters}); got "
            f"[{cluster_flat.min()}, {cluster_flat.max()}].")
    if column_flat.min() < 0 or column_flat.max() >= page_group_size:
        raise ValueError(
            f"{subject} columns out of range [0, {page_group_size}); got "
            f"[{column_flat.min()}, {column_flat.max()}].")
    slots = cluster_flat * page_group_size + column_flat
    if np.unique(slots).size != cluster_flat.size:
        raise ValueError(
            f"{subject} is not a bijection: some (cluster, column) slot is "
            "shared or unused.")
    counts = np.bincount(cluster_flat, minlength=num_clusters)
    if not bool(np.all(counts == page_group_size)):
        raise ValueError(
            f"{subject} clusters are not all full to "
            f"page_group_size={page_group_size}; member counts seen: "
            f"{sorted(set(counts.tolist()))}.")


def load_cluster_map(
    path: str,
    page_group_size: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and validate a head-group cluster map ``.npz``.

    Returns ``(cluster_of, column_of)`` as int64 CPU tensors
    ``[num_layers, num_kv_heads]``, ``num_layers`` read from the map. Validates
    the keys, the map's ``page_group_size`` and ``num_kv_heads`` against the
    runtime, and the bijection: every (layer, head) occupies a distinct column
    and every cluster is full.
    """
    import numpy as np

    data = np.load(path, allow_pickle=False)
    for key in ("cluster_of", "column_of", "page_group_size"):
        if key not in data:
            raise ValueError(
                f"head_group_cluster_map {path!r} is missing key {key!r}; "
                f"present keys: {list(data.files)}.")
    map_page_group_size = int(data["page_group_size"])
    if map_page_group_size != page_group_size:
        raise ValueError(
            f"head_group_cluster_map page_group_size ({map_page_group_size}) "
            f"!= runtime page_group_size ({page_group_size}).")
    cluster_of = data["cluster_of"]
    column_of = data["column_of"]
    if cluster_of.ndim != 2 or cluster_of.shape != column_of.shape:
        raise ValueError(
            f"head_group_cluster_map cluster_of/column_of must be matching 2-D "
            f"[num_layers, num_kv_heads]; got {cluster_of.shape} / "
            f"{column_of.shape}.")
    num_layers, map_num_kv_heads = cluster_of.shape
    if map_num_kv_heads != num_kv_heads:
        raise ValueError(
            f"head_group_cluster_map has {map_num_kv_heads} KV heads per layer "
            f"but the model has {num_kv_heads}.")
    num_members = num_layers * num_kv_heads
    num_clusters = num_members // page_group_size
    cluster_flat = cluster_of.reshape(-1).astype(np.int64)
    column_flat = column_of.reshape(-1).astype(np.int64)
    _validate_bijection(
        cluster_flat, column_flat,
        num_clusters=num_clusters,
        page_group_size=page_group_size,
        subject="head_group_cluster_map")
    return (
        torch.from_numpy(cluster_flat).view(num_layers, num_kv_heads),
        torch.from_numpy(column_flat).view(num_layers, num_kv_heads),
    )


def read_cluster_map_meta(path: str) -> dict | None:
    """Return a cluster map's ``meta`` JSON (provenance: ``source_model``,
    ``page_group_size``, ``cluster_scope``, ``static_layer_ids``, ...), or
    ``None`` for older maps without it. Sole reader of that blob.
    """
    import json

    import numpy as np

    data = np.load(path, allow_pickle=False)
    if "meta" not in data:
        return None
    return json.loads(bytes(data["meta"]).decode("utf-8"))


def read_cluster_map_static_layer_ids(path: str) -> list[int] | None:
    """The ``static_layer_ids`` a cluster map records, or ``None`` for a dense
    or older map without the field. A hybrid map is authored over the
    full-attention layers only, so recording their physical indices lets a
    consumer assert the map matches the running model's layers, not just their
    count.
    """
    meta = read_cluster_map_meta(path)
    if meta is None:
        return None
    static_layer_ids = meta.get("static_layer_ids")
    if static_layer_ids is None:
        return None
    return [int(x) for x in static_layer_ids]


def member_maps_from_cluster_map(
    cluster_of: torch.Tensor,
    column_of: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat ``(member_to_cluster, member_to_col)`` from a cluster map.

    Flattening the ``[num_layers, num_kv_heads]`` map row-major gives member
    row ``m = layer * num_kv_heads + head``, matching ``identity_member_maps``.
    """
    member_to_cluster = cluster_of.reshape(-1).to(torch.int64)
    member_to_col = column_of.reshape(-1).to(torch.int64)
    return member_to_cluster, member_to_col


def physical_member_maps_from_static_cluster_map(
    static_cluster_of: torch.Tensor,
    static_column_of: torch.Tensor,
    static_layer_ids: list[int],
    num_layers: int,
    page_group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a static (full-attention-only) cluster map to the physical layout.

    A map is authored over compressible layers only, but every layer physically
    stores KV, so the ragged builder needs ``[num_layers, num_kv_heads]``.

    Static layers land at ``R(c) = static_layer_ids[c // ng] * ng + (c % ng)``,
    ``ng = num_kv_heads // page_group_size`` -- the same block-table row
    ``executor.run_request`` reorders for compressor cluster ``c``. Columns are
    copied verbatim, so the executor's per-column head ordering matches the
    physical page. Sliding layers get identity grouping on their own rows.

    The two sets are disjoint, being different physical layers, and must stay
    so: a shared cluster would let compression truncate a sliding layer's
    uncompressed KV.
    """
    num_static, num_kv_heads = static_cluster_of.shape
    if len(static_layer_ids) != num_static:
        raise ValueError(
            f"static_layer_ids has {len(static_layer_ids)} entries but the "
            f"cluster map spans {num_static} static layers.")
    if num_kv_heads % page_group_size != 0:
        raise ValueError(
            f"num_kv_heads ({num_kv_heads}) not divisible by page_group_size "
            f"({page_group_size}).")
    ng = num_kv_heads // page_group_size
    device = static_cluster_of.device
    cluster_of = torch.empty(
        (num_layers, num_kv_heads), dtype=torch.int64, device=device)
    column_of = torch.empty(
        (num_layers, num_kv_heads), dtype=torch.int64, device=device)

    heads = torch.arange(num_kv_heads, dtype=torch.int64, device=device)
    ident_cluster = torch.div(heads, page_group_size, rounding_mode="floor")
    ident_col = heads % page_group_size
    static_set = {int(x) for x in static_layer_ids}
    for layer in range(num_layers):
        if layer not in static_set:
            cluster_of[layer] = layer * ng + ident_cluster
            column_of[layer] = ident_col

    sids = torch.as_tensor(static_layer_ids, dtype=torch.int64, device=device)
    for static_idx, layer in enumerate(static_layer_ids):
        c = static_cluster_of[static_idx].to(torch.int64)
        phys_cluster = (
            sids[torch.div(c, ng, rounding_mode="floor")] * ng + (c % ng))
        cluster_of[layer] = phys_cluster
        column_of[layer] = static_column_of[static_idx].to(torch.int64)

    _validate_bijection(
        cluster_of.reshape(-1).to(torch.int64).cpu().numpy(),
        column_of.reshape(-1).to(torch.int64).cpu().numpy(),
        num_clusters=num_layers * num_kv_heads // page_group_size,
        page_group_size=page_group_size,
        subject="derived physical cluster map")
    return cluster_of, column_of


def member_virtual_block_table(
    physical_block_table: torch.Tensor,
    member_to_cluster: torch.Tensor,
    member_to_col: torch.Tensor,
    page_group_size: int,
    cluster_axis: int,
) -> torch.Tensor:
    """Gather each member's cluster, then fold its column into
    ``physical_block * page_group_size + column``; the cluster axis becomes the
    member axis. Dtype is preserved: FlashAttention requires int32.
    """
    gathered = physical_block_table.index_select(cluster_axis, member_to_cluster)
    columns = _broadcast_along(
        member_to_col, gathered.ndim, cluster_axis).to(gathered.dtype)
    return gathered * page_group_size + columns


def member_virtual_slots(
    group_slots: torch.Tensor,
    member_to_cluster: torch.Tensor,
    member_to_col: torch.Tensor,
    page_group_size: int,
    block_size: int,
    cluster_axis: int,
) -> torch.Tensor:
    """Gather each member's cluster and re-encode the physical slot
    ``block * block_size + offset`` into its column, as
    ``(block * page_group_size + column) * block_size + offset``.
    """
    gathered = group_slots.index_select(cluster_axis, member_to_cluster)
    block = torch.div(gathered, block_size, rounding_mode="floor")
    offset = gathered - block * block_size
    columns = _broadcast_along(
        member_to_col, gathered.ndim, cluster_axis).to(gathered.dtype)
    return (block * page_group_size + columns) * block_size + offset


def member_seq_lens(
    group_seq_lens: torch.Tensor,
    member_to_cluster: torch.Tensor,
    cluster_axis: int,
) -> torch.Tensor:
    """Per-member sequence lengths by gathering each member's cluster.

    A gather, not a computation: members share their cluster's single length
    because they share its physical blocks, under compression too.
    """
    return group_seq_lens.index_select(cluster_axis, member_to_cluster)


@dataclass(frozen=True)
class RaggedStepViews:
    """Per-step ragged-paging views consumed by the attention forward path.

    Every non-scalar field puts the cluster/member axis first, so the forward
    path slices one layer without a transpose.

    ``ragged_decode_layout`` is set when every sequence has query length 1, and
    then ``seq_lens_grouped`` / ``slot_mapping_grouped`` carry the
    ``[num_layers, num_reqs, num_kv_heads]`` decode overlay rather than the
    member-major ``[num_members_total, ...]`` prefill/mixed layout.
    ``num_head_groups_per_layer`` and ``page_group_size`` are the layout
    constants the forward path needs alongside the views: they are fixed for the
    model, and travel here so the attention metadata carries one ragged field
    rather than a dozen.
    ``cluster_block_table`` is ``[num_reqs, num_clusters_total, max_blocks]``
    trimmed to what this batch occupies, ``clusters_per_layer`` and
    ``cols_per_layer`` are the ``[num_layers, num_kv_heads]`` member maps, and
    ``query_start_loc_grouped`` holds cumulative query lengths per layer's
    member sequences -- the layer slice happens in the forward path.
    """

    num_layers_local: int
    num_head_groups_per_layer: int
    page_group_size: int
    ragged_decode_layout: bool
    cluster_block_table: torch.Tensor
    clusters_per_layer: torch.Tensor
    cols_per_layer: torch.Tensor
    seq_lens_grouped: torch.Tensor
    slot_mapping_grouped: torch.Tensor
    query_start_loc_grouped: torch.Tensor


def layer_overlay(
    metadata,
    *,
    num_actual_tokens: int,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
):
    """A per-layer view of one step's attention metadata.

    Under ragged paging a layer's sequences are its own (request, KV-head)
    members, so five fields differ per layer while every other field of the
    step's metadata is shared. This returns a shallow copy with exactly those
    five replaced -- the single place that says which five they are.

    ``copy.copy`` rather than ``dataclasses.replace``: replace() re-runs
    ``__init__`` over every field, which is pure-Python work paid once per layer
    per step on the eager critical path. Copying ``__dict__`` and overwriting
    five entries leaves the rest as shared read-only references, which is what
    the caller wants anyway. Equivalent because the metadata dataclasses on this
    path define no ``__post_init__``.
    """
    overlay = copy.copy(metadata)
    overlay.num_actual_tokens = num_actual_tokens
    overlay.block_table = block_table
    overlay.seq_lens = seq_lens
    overlay.slot_mapping = slot_mapping
    overlay.query_start_loc = query_start_loc
    return overlay


def build_decode_layer_overlays(
    metadata,
    views: "RaggedStepViews",
    *,
    num_reqs: int,
    num_kv_heads_per_layer: int,
) -> list:
    """One metadata overlay per layer, for the uniform-decode layout.

    Uniform decode is one query token per (request, KV-head) member, so a
    layer's shapes are known as soon as the step's views are: building all of
    them up front lets the attention forward pick its layer with a list lookup
    instead of assembling an overlay inside the per-layer call.

    Only valid when ``views.ragged_decode_layout`` is set -- that is what puts
    ``seq_lens_grouped`` / ``slot_mapping_grouped`` in the
    ``[num_layers, num_reqs, num_kv_heads]`` form indexed here.
    """
    assert views.ragged_decode_layout, (
        "decode overlays need the uniform-decode layout; the member-major "
        "layout builds its overlay per call instead.")
    num_virtual_seqs = num_reqs * num_kv_heads_per_layer
    overlays = []
    for layer_idx in range(views.num_layers_local):
        # This layer's virtual block table, built on demand:
        # [num_reqs, num_kv_heads, max_blocks] ->
        # [num_reqs * num_kv_heads, max_blocks].
        block_table_layer = member_virtual_block_table(
            views.cluster_block_table,
            views.clusters_per_layer[layer_idx],
            views.cols_per_layer[layer_idx],
            views.page_group_size,
            cluster_axis=1,
        ).reshape(num_virtual_seqs, -1)
        overlays.append(layer_overlay(
            metadata,
            num_actual_tokens=num_virtual_seqs,
            block_table=block_table_layer,
            seq_lens=views.seq_lens_grouped[layer_idx].view(num_virtual_seqs),
            slot_mapping=views.slot_mapping_grouped[layer_idx].reshape(-1),
            query_start_loc=views.query_start_loc_grouped,
        ))
    return overlays


def build_ragged_step_views(
    *,
    block_table_tensor: torch.Tensor,
    slot_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    effective_seq_lens_cpu,
    num_reqs: int,
    num_actual_tokens: int,
    max_query_len: int,
    max_seq_len: int,
    num_head_groups_per_layer: int,
    page_group_size: int,
    block_size: int,
    num_kv_heads_per_layer: int,
    member_maps_fn: "Callable[[int, torch.device], tuple[torch.Tensor, torch.Tensor]]",
) -> RaggedStepViews:
    """Build the ragged-paging per-step views for one attention-metadata build.

    Backend-agnostic: a second attention backend can call this and wrap the
    result in its own metadata object. ``member_maps_fn(num_layers_local,
    device)`` returns ``(member_to_cluster, member_to_col)``, passed in rather
    than computed here because resolving it depends on the caller's cached
    cluster-map state.

    ``effective_seq_lens_cpu`` is the post-compression per-cluster length,
    ``[num_reqs, num_clusters_total]``, or None with compression off; when set
    the cache holds ``effective[cluster]`` tokens plus this step's chunk.

    Raises RuntimeError when the incoming shapes or the uniform-decode
    invariant do not match the ragged layout.
    """
    import numpy as np

    from vllm.utils.math_utils import cdiv

    # The block table stacks every layer's clusters on axis 1.
    if block_table_tensor.ndim != 3:
        raise RuntimeError(
            "ragged path expects 3D block_table_tensor "
            "[num_reqs, num_head_groups_total, max_blocks_per_req], "
            f"got shape {tuple(block_table_tensor.shape)}.")
    num_head_groups_total = block_table_tensor.shape[1]
    if num_head_groups_total % num_head_groups_per_layer != 0:
        raise RuntimeError(
            f"num_head_groups_total ({num_head_groups_total}) is not "
            "divisible by num_head_groups_per_layer "
            f"({num_head_groups_per_layer}).")
    num_layers_local = num_head_groups_total // num_head_groups_per_layer
    if slot_mapping.ndim != 2:
        raise RuntimeError(
            "ragged path expects 2D slot_mapping "
            "[num_head_groups_total, num_actual_tokens], "
            f"got shape {tuple(slot_mapping.shape)}.")
    ragged_decode_layout = max_query_len == 1

    # Per-cluster lengths, [num_clusters_total, num_reqs]. The decode overlay
    # is a reshape of this, so both layouts share one construction.
    if effective_seq_lens_cpu is not None:
        query_start_loc_cpu = query_start_loc_cpu[: num_reqs + 1]
        num_scheduled_np = (
            query_start_loc_cpu[1:].cpu().numpy()
            - query_start_loc_cpu[:-1].cpu().numpy()
        )
        effective_np = np.asarray(effective_seq_lens_cpu, dtype=np.int32)
        seq_lens_cluster_np = (
            effective_np[:num_reqs].T.astype(np.int32)
            + num_scheduled_np.astype(np.int32)[None, :]
        )
        seq_lens_cluster = torch.from_numpy(
            seq_lens_cluster_np).to(seq_lens.device).contiguous()
    else:
        seq_lens_cluster = (
            seq_lens.unsqueeze(0)
            .expand(num_head_groups_total, -1)
            .contiguous()
        )

    # Gathering on the FLAT cluster axis, before the per-layer reshape, is what
    # lets a cluster span layers.
    member_to_cluster, member_to_col = member_maps_fn(
        num_layers_local, seq_lens.device)
    # Each attention call builds its virtual block table from these.
    max_blocks = cdiv(max_seq_len, block_size)
    cluster_block_table = block_table_tensor[:, :, :max_blocks]
    clusters_per_layer = member_to_cluster.view(
        num_layers_local, num_kv_heads_per_layer)
    cols_per_layer = member_to_col.view(
        num_layers_local, num_kv_heads_per_layer)
    # These member views carry no block axis, so the all-member form is cheap.
    slot_mapping_member = member_virtual_slots(
        slot_mapping, member_to_cluster, member_to_col,
        page_group_size, block_size, cluster_axis=0)
    # seq_lens_cluster: [num_clusters_total, num_reqs].
    seq_lens_member = member_seq_lens(
        seq_lens_cluster, member_to_cluster, cluster_axis=0)

    if ragged_decode_layout:
        # Uniform decode, so num_actual_tokens == num_reqs.
        if num_actual_tokens != num_reqs:
            raise RuntimeError(
                "uniform-decode ragged path expects "
                f"num_actual_tokens ({num_actual_tokens}) == num_reqs "
                f"({num_reqs}).")
        # [num_members_total, num_reqs] -> [num_layers, num_reqs, num_kv_heads].
        slot_mapping_grouped = (
            slot_mapping_member
            .view(num_layers_local, num_kv_heads_per_layer, num_reqs)
            .permute(0, 2, 1)
            .contiguous()
        )
        seq_lens_grouped = (
            seq_lens_member
            .view(num_layers_local, num_kv_heads_per_layer, num_reqs)
            .permute(0, 2, 1)
            .contiguous()
        )
        # Length 1 each, so cu_seqlens is a plain arange.
        query_start_loc_grouped = torch.arange(
            num_kv_heads_per_layer * num_reqs + 1,
            device=query_start_loc.device,
            dtype=query_start_loc.dtype,
        )
    else:
        # Member-major (prefill / mixed): the forward path's copy is required
        # because (req, member) sequences are contiguous in no token-major
        # view when lengths vary.
        slot_mapping_grouped = slot_mapping_member
        seq_lens_grouped = seq_lens_member
        # cu_seqlens per layer's member sequences; the forward path slices.
        query_start_head = query_start_loc[:num_reqs]
        offsets = torch.arange(
            num_kv_heads_per_layer,
            device=query_start_head.device,
            dtype=query_start_head.dtype,
        ) * num_actual_tokens
        query_start_grouped_head = (
            query_start_head.unsqueeze(0) + offsets.unsqueeze(1)
        ).reshape(-1)
        query_start_grouped_tail = torch.tensor(
            [num_kv_heads_per_layer * num_actual_tokens],
            device=query_start_head.device,
            dtype=query_start_head.dtype,
        )
        query_start_loc_grouped = torch.cat(
            [query_start_grouped_head, query_start_grouped_tail])

    return RaggedStepViews(
        num_layers_local=num_layers_local,
        num_head_groups_per_layer=num_head_groups_per_layer,
        page_group_size=page_group_size,
        ragged_decode_layout=ragged_decode_layout,
        cluster_block_table=cluster_block_table,
        clusters_per_layer=clusters_per_layer,
        cols_per_layer=cols_per_layer,
        seq_lens_grouped=seq_lens_grouped,
        slot_mapping_grouped=slot_mapping_grouped,
        query_start_loc_grouped=query_start_loc_grouped,
    )
