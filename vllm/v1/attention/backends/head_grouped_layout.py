# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Column-major head-grouped KV cache layout and virtual-block addressing.

Single source of truth for the column-major page layout used by head-grouped
paging. Allocation, the FlashAttention metadata builder, and the attention
forward path all address the cache through the helpers here, so the column-major
/ virtual-block arithmetic lives in exactly one place. Swapping the identity
cluster map for a real (cross-layer) cluster map only changes the
member->(cluster, column) mapping, not these helpers.

Layout
------
A head-grouped KV cache page is stored **column-major**: the per-page column
dimension (``page_group_size``) sits OUTSIDE ``block_size``::

    [2, num_physical_blocks, page_group_size, block_size, head_size]
         ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^  ^^^^^^^^^^
         physical blocks      column            tokens in block

so each ``(physical_block, column)`` pair is a fully contiguous standard
single-head block, addressable by a flat **virtual block id**::

    virtual_block_id = physical_block * page_group_size + column

Flattening ``(num_physical_blocks, page_group_size)`` yields the virtual-block
axis, so the cache viewed as::

    [2, num_physical_blocks * page_group_size, block_size, 1, head_size]

is the standard single-KV-head paged layout that FlashAttention consumes with no
kernel change and full coalescing. A head-group cluster owns a set of physical
blocks (shared retention budget); its members are separated by their column.

Identity cluster map
--------------------
With the identity map the layout reproduces adjacent-head grouping: KV head
``m`` of a layer occupies cluster ``m // page_group_size`` at column
``m % page_group_size``. Members of one cluster share the same physical blocks at
columns ``0 .. page_group_size - 1``.
"""
from __future__ import annotations

import torch


def column_major_cache_shape(
    num_blocks: int,
    block_size: int,
    page_group_size: int,
    head_size: int,
) -> tuple[int, int, int, int, int]:
    """Physical shape of a column-major head-grouped KV cache tensor.

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
        f"column-major head-grouped cache must lead with the key/value axis "
        f"of size 2; got shape {tuple(kv_cache.shape)}.")
    return kv_cache.reshape(
        2, num_blocks * page_group_size, block_size, 1, head_size)


# --- Identity-map addressing API (public; for cluster-map tooling/tests) -----
#
# The helpers in this section address the cache assuming the IDENTITY cluster
# map (adjacent-head grouping: KV head ``m`` -> cluster ``m // page_group_size``
# at column ``m % page_group_size``). They are intentionally kept as a stable,
# self-contained public API for external head-group cluster-map tooling and for
# the layout test suite (``tests/v1/attention/test_head_grouped_layout.py``),
# which exercises the identity case directly as the reference for the general
# mapping below.
#
# They are NOT on the live engine path. The runtime addresses members through
# the explicit cluster-map-driven helpers further down
# (``member_virtual_block_table`` / ``member_virtual_slots`` /
# ``member_seq_lens``), which subsume the identity map as one instance. Keep the
# two paths numerically consistent: any change here must hold for the
# cluster-map helpers under an identity map, and vice versa.


def identity_member_columns(
    num_kv_heads: int,
    page_group_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Column of each layer-local KV head under the identity cluster map.

    KV head ``m`` lives at column ``m % page_group_size``. Shape
    ``[num_kv_heads]``.
    """
    members = torch.arange(num_kv_heads, dtype=torch.int64, device=device)
    return members % page_group_size


def identity_member_clusters(
    num_kv_heads: int,
    page_group_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Cluster (head-group) of each layer-local KV head under the identity map.

    KV head ``m`` belongs to cluster ``m // page_group_size``. Shape
    ``[num_kv_heads]``.
    """
    members = torch.arange(num_kv_heads, dtype=torch.int64, device=device)
    return members // page_group_size


def _identity_member_columns_along(
    num_clusters: int,
    page_group_size: int,
    ndim: int,
    group_axis: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Column index of each member along an expanded cluster axis.

    The cluster axis of length ``num_clusters`` expands to ``num_clusters *
    page_group_size`` members ordered member-major (cluster outer, column
    inner): member position ``i`` is cluster ``i // page_group_size`` at column
    ``i % page_group_size``. Returns the columns broadcast-shaped so they align
    with ``group_axis`` of an ``ndim`` tensor (size 1 on every other axis).
    """
    columns = torch.arange(
        num_clusters * page_group_size, device=device, dtype=torch.int64
    ) % page_group_size
    shape = [1] * ndim
    shape[group_axis] = num_clusters * page_group_size
    return columns.view(shape)


def physical_to_virtual_slots(
    group_slots: torch.Tensor,
    page_group_size: int,
    block_size: int,
    group_axis: int = 0,
) -> torch.Tensor:
    """Expand per-cluster physical write slots into per-member virtual slots.

    ``group_slots`` holds the physical slot ``physical_block * block_size +
    offset`` for each cluster along ``group_axis`` (identity cluster map, so a
    cluster's members are contiguous columns ``0 .. page_group_size - 1``). The
    cluster axis expands by ``page_group_size`` into members ordered
    member-major; member ``(cluster, column)`` writes the SAME token at its own
    column, so its virtual slot is::

        virtual_slot = (physical_block * page_group_size + column) * block_size
                       + offset

    Returns the tensor with ``group_axis`` grown to ``num_clusters *
    page_group_size``, so the member position along that axis is the
    layer-local KV head index.
    """
    num_clusters = group_slots.shape[group_axis]
    physical_block = torch.div(group_slots, block_size, rounding_mode="floor")
    offset = group_slots - physical_block * block_size
    physical_block = physical_block.repeat_interleave(
        page_group_size, dim=group_axis)
    offset = offset.repeat_interleave(page_group_size, dim=group_axis)
    columns = _identity_member_columns_along(
        num_clusters, page_group_size, group_slots.ndim, group_axis,
        group_slots.device)
    virtual_block = physical_block * page_group_size + columns
    return virtual_block * block_size + offset


def physical_to_virtual_block_table(
    group_block_table: torch.Tensor,
    page_group_size: int,
    group_axis: int = 0,
) -> torch.Tensor:
    """Expand a per-cluster physical block table into per-member virtual ids.

    ``group_block_table`` holds physical block ids with the cluster axis at
    ``group_axis`` (identity cluster map). The cluster axis expands by
    ``page_group_size`` into members ordered member-major; member ``(cluster,
    column)`` reads the cluster's physical blocks at its own column, so each
    entry becomes::

        virtual_block = physical_block * page_group_size + column

    Returns the tensor with ``group_axis`` grown to ``num_clusters *
    page_group_size`` (member position = layer-local KV head index). The
    block-id dtype is preserved (FlashAttention requires int32 block tables).
    """
    num_clusters = group_block_table.shape[group_axis]
    virtual = group_block_table.repeat_interleave(
        page_group_size, dim=group_axis) * page_group_size
    columns = _identity_member_columns_along(
        num_clusters, page_group_size, group_block_table.ndim, group_axis,
        group_block_table.device).to(virtual.dtype)
    return virtual + columns


def expand_member_seq_lens(
    group_seq_lens: torch.Tensor,
    page_group_size: int,
    group_axis: int = 0,
) -> torch.Tensor:
    """Broadcast per-cluster sequence lengths to per-member.

    Under the identity cluster map every member of a cluster shares the
    cluster's (max-pooled) retention length, so the cluster axis is simply
    repeated ``page_group_size`` times in member-major order. This sharing is
    invariant: a cluster's members occupy the same physical KV blocks, so they
    always carry one shared length (the cluster map changes only which heads are
    pooled into a cluster, never giving members individual lengths).
    """
    return group_seq_lens.repeat_interleave(page_group_size, dim=group_axis)


# --- Cluster-map-driven member addressing ----------------------------------
#
# The identity helpers above expand each cluster into ``page_group_size``
# CONTIGUOUS members (adjacent-head grouping). A real cluster map instead
# assigns each KV head ``(layer, head)`` to an arbitrary ``(cluster, column)``
# — clusters may even span layers. The functions below address members through
# explicit flat index tensors so the identity map is just one instance of the
# general mechanism:
#
#   member row  m = layer * num_kv_heads + head   (layer-contiguous, member-major)
#   member_to_cluster[m] -> cluster id that head reads/writes
#   member_to_col[m]     -> column of that head within the cluster's page
#
# A member's KV lives in its cluster's physical blocks at its column, folded
# into virtual block ids ``physical_block * page_group_size + column``. Only the
# physical placement changes, so at no compression the attention output is
# independent of the map.


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
    """Identity cluster map as flat ``(member_to_cluster, member_to_col)``.

    Head ``h`` of layer ``L`` -> cluster ``L * (num_kv_heads // page_group_size)
    + h // page_group_size`` at column ``h % page_group_size`` (adjacent-head
    grouping). Member row ``m = L * num_kv_heads + h``. Both
    tensors are shape ``[num_layers * num_kv_heads]``; ``member_to_cluster`` is
    int64 (used for ``index_select``).
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


def load_cluster_map(
    path: str,
    page_group_size: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and validate a head-group cluster map ``.npz``.

    Returns ``(cluster_of, column_of)`` as int64 CPU tensors of shape
    ``[num_layers, num_kv_heads]`` (``num_layers`` is read from the map, which is
    built by ``tools/head_group_clustering``). Validates the keys, the
    map's ``page_group_size`` against the runtime value, ``num_kv_heads``, and
    the (cluster, column) bijection: every (layer, head) occupies a distinct
    column of a cluster and every cluster is full to ``page_group_size``.
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
    if cluster_flat.min() < 0 or cluster_flat.max() >= num_clusters:
        raise ValueError(
            f"head_group_cluster_map cluster ids out of range [0, "
            f"{num_clusters}); got [{cluster_flat.min()}, "
            f"{cluster_flat.max()}].")
    if column_flat.min() < 0 or column_flat.max() >= page_group_size:
        raise ValueError(
            f"head_group_cluster_map columns out of range [0, "
            f"{page_group_size}); got [{column_flat.min()}, "
            f"{column_flat.max()}].")
    # Bijection: each (cluster, column) slot is occupied exactly once and every
    # cluster is full, so the heads partition the clusters evenly.
    slots = cluster_flat * page_group_size + column_flat
    if np.unique(slots).size != num_members:
        raise ValueError(
            "head_group_cluster_map is not a bijection: some (cluster, column) "
            "slot is shared or unused.")
    counts = np.bincount(cluster_flat, minlength=num_clusters)
    if not bool(np.all(counts == page_group_size)):
        raise ValueError(
            f"head_group_cluster_map clusters are not all full to "
            f"page_group_size={page_group_size}; member counts seen: "
            f"{sorted(set(counts.tolist()))}.")
    return (
        torch.from_numpy(cluster_flat).view(num_layers, num_kv_heads),
        torch.from_numpy(column_flat).view(num_layers, num_kv_heads),
    )


def read_cluster_map_static_layer_ids(path: str) -> list[int] | None:
    """Return the ``static_layer_ids`` a cluster map records in its metadata,
    or ``None`` if absent.

    A sliding-window hybrid cluster map is authored over the full-attention
    layers only; ``tools/head_group_clustering/build_cluster_map`` records the
    physical indices of those layers so a consumer can assert the map matches
    the running model's full-attention layers (not just their count). Dense
    maps and older maps without the field return ``None``.
    """
    import json

    import numpy as np

    data = np.load(path, allow_pickle=False)
    if "meta" not in data:
        return None
    meta = json.loads(bytes(data["meta"]).decode("utf-8"))
    static_layer_ids = meta.get("static_layer_ids")
    if static_layer_ids is None:
        return None
    return [int(x) for x in static_layer_ids]


def member_maps_from_cluster_map(
    cluster_of: torch.Tensor,
    column_of: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat ``(member_to_cluster, member_to_col)`` from a cluster map.

    ``cluster_of`` / ``column_of`` are ``[num_layers, num_kv_heads]`` as built by
    ``tools/head_group_clustering``. Flattening row-major gives member row
    ``m = layer * num_kv_heads + head``, matching ``identity_member_maps``.
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

    Sliding-window hybrid models compress only their full-attention layers, so
    the cluster map is authored over those layers alone — shape
    ``[num_static, num_kv_heads]``. Physically every layer stores KV, so the
    FlashAttention head-grouped builder needs a ``[num_layers, num_kv_heads]``
    assignment. This expansion produces it:

    * **static layers** — each full-attention head's KV is placed at physical
      cluster ``R(c) = static_layer_ids[c // ng] * ng + (c % ng)``, the SAME
      block-table row the compression executor reorders for compressor cluster
      ``c`` (``executor.run_request``: ``layer_idx * num_groups + group`` with
      ``layer_idx = compressed_layer_ids[static_idx]``). ``ng = num_kv_heads //
      page_group_size``. The column is copied verbatim, so the executor's
      per-column head ordering matches the physical page exactly.
    * **sliding layers** — identity (adjacent-head) grouping at the layer's own
      cluster rows, disjoint from the static rows because static and sliding
      layers are different physical layers.

    The result is a valid global bijection over ``num_layers * num_kv_heads``
    cells into ``num_layers * ng`` full clusters. Static and sliding heads never
    share a cluster — a shared cluster would let compression truncate a
    sliding layer's (uncompressed) full KV. Returns ``(cluster_of, column_of)``
    of shape ``[num_layers, num_kv_heads]``, int64, on the inputs' device.
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

    _validate_member_bijection(
        cluster_of, column_of, num_layers, num_kv_heads, page_group_size)
    return cluster_of, column_of


def _validate_member_bijection(
    cluster_of: torch.Tensor,
    column_of: torch.Tensor,
    num_layers: int,
    num_kv_heads: int,
    page_group_size: int,
) -> None:
    """Assert ``(cluster_of, column_of)`` is a valid (cluster, column)
    bijection: every slot occupied once, every cluster full. Mirrors the file
    checks in ``load_cluster_map`` for engine-derived (not file-loaded) maps."""
    import numpy as np

    num_clusters = num_layers * num_kv_heads // page_group_size
    co = cluster_of.reshape(-1).cpu().numpy().astype(np.int64)
    col = column_of.reshape(-1).cpu().numpy().astype(np.int64)
    if co.min() < 0 or co.max() >= num_clusters:
        raise ValueError(
            f"derived cluster ids out of range [0, {num_clusters}); got "
            f"[{co.min()}, {co.max()}].")
    if col.min() < 0 or col.max() >= page_group_size:
        raise ValueError(
            f"derived columns out of range [0, {page_group_size}); got "
            f"[{col.min()}, {col.max()}].")
    slots = co * page_group_size + col
    if np.unique(slots).size != co.size:
        raise ValueError(
            "derived physical cluster map is not a bijection: a (cluster, "
            "column) slot is shared or unused.")
    counts = np.bincount(co, minlength=num_clusters)
    if not bool(np.all(counts == page_group_size)):
        raise ValueError(
            "derived physical cluster map clusters are not all full to "
            f"page_group_size={page_group_size}.")


def member_virtual_block_table(
    physical_block_table: torch.Tensor,
    member_to_cluster: torch.Tensor,
    member_to_col: torch.Tensor,
    page_group_size: int,
    cluster_axis: int,
) -> torch.Tensor:
    """Per-member virtual block table by gathering each member's cluster.

    ``physical_block_table`` has the cluster axis at ``cluster_axis`` (length
    ``num_clusters``). Gathers the cluster of each member, then folds the
    member's column into the virtual block id ``physical_block *
    page_group_size + column``. The cluster axis becomes the member axis
    (length ``len(member_to_cluster)``). Block-id dtype is preserved (FA
    requires int32).
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
    """Per-member virtual write slots by gathering each member's cluster.

    ``group_slots`` holds the physical slot ``physical_block * block_size +
    offset`` with the cluster axis at ``cluster_axis``. Gathers each member's
    cluster and re-encodes the slot into the member's column::

        virtual_slot = (physical_block * page_group_size + column) * block_size
                       + offset
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

    Every member of a cluster shares the cluster's single (max-pooled) length
    because they share the cluster's physical KV blocks; this gather just
    broadcasts that shared length to each member. The sharing holds under
    compression too: one length is kept per cluster, the cluster map only
    changing which heads form the cluster.
    """
    return group_seq_lens.index_select(cluster_axis, member_to_cluster)
