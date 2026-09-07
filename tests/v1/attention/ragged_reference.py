# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity-cluster-map addressing, kept as the reference the engine is tested against.

The engine addresses the cache through the cluster-map-driven helpers in
``vllm/v1/attention/backends/ragged_layout.py``, which take explicit
``member_to_cluster`` / ``member_to_col`` tensors. This module spells out the
same arithmetic for the one map those helpers subsume -- the identity map,
where KV head ``m`` is cluster ``m // page_group_size`` at column
``m % page_group_size``.

Two independent derivations of one addressing rule: a change to the engine
helpers must keep agreeing with these under an identity map, which is what
``test_ragged_layout.py`` and ``test_ragged_custom_op.py`` check. Nothing in
``vllm/`` imports this module.
"""
import torch


def identity_member_columns(
    num_kv_heads: int,
    page_group_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``[num_kv_heads]``: KV head ``m`` sits at column
    ``m % page_group_size``."""
    members = torch.arange(num_kv_heads, dtype=torch.int64, device=device)
    return members % page_group_size


def identity_member_clusters(
    num_kv_heads: int,
    page_group_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """``[num_kv_heads]``: KV head ``m`` belongs to cluster
    ``m // page_group_size``."""
    members = torch.arange(num_kv_heads, dtype=torch.int64, device=device)
    return members // page_group_size


def _identity_member_columns_along(
    num_clusters: int,
    page_group_size: int,
    ndim: int,
    group_axis: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Member-major columns for an expanded cluster axis: member ``i`` is
    cluster ``i // page_group_size`` at column ``i % page_group_size``. Shaped
    to broadcast along ``group_axis`` of an ``ndim`` tensor.
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

    Each member writes the SAME token at its own column, so a physical
    ``block * block_size + offset`` becomes
    ``(block * page_group_size + column) * block_size + offset``.
    ``group_axis`` grows member-major, its position then being the layer-local
    KV head index.
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
    """Expand a per-cluster physical block table into per-member virtual ids
    ``physical_block * page_group_size + column``, ``group_axis`` growing
    member-major. Dtype is preserved: FlashAttention requires int32.
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
    """Repeat the cluster axis ``page_group_size`` times, member-major. The
    sharing is invariant, not an identity-map artefact: a cluster's members
    occupy the same blocks, so they always carry one length.
    """
    return group_seq_lens.repeat_interleave(page_group_size, dim=group_axis)
