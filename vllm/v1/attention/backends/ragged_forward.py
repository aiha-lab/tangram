# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ragged-paging attention forward: token-major tensors to member-major and back.

Column-major paging makes every KV head its own varlen sequence, so a
(request, KV-head) pair -- a member -- is one sequence carrying its GQA group
of query heads. The attention backend is handed member-major tensors and
per-layer metadata; producing them is this module's whole job, and it is
separate from ``vllm/attention/layer.py`` because that file is upstream-generic
while this reshape is specific to the ragged v1 backend.

Two paths, chosen by ``FlashAttentionMetadata.ragged_decode_layout``. Uniform
decode is one query token per member, so token-major flattens into member-major
by reshape alone. Prefill and mixed batches have varying sequence lengths, for
which no single view of token-major Q/K/V is member-contiguous, so they copy.

``_ragged_attention_impl`` is the body of the ``vllm::unified_attention_ragged``
custom op and always runs eagerly between captured CUDA-graph pieces, which is
why its inputs may be padded to a capture size.
"""
# The ``Attention`` annotations below name a class in vllm/attention/layer.py,
# which imports this module: deferring annotation evaluation is what keeps that
# import one-way, with no runtime dependency back on the layer.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backends.ragged_layout import (
    layer_overlay,
    member_virtual_block_table,
)

if TYPE_CHECKING:
    from vllm.attention.layer import Attention


def _ragged_attention_impl(
    layer: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    attn_metadata,
    kv_cache: torch.Tensor,
) -> None:
    """Ragged attention body, writing into ``output`` in place.

    Each (request, KV-head member) is one varlen sequence: column-major paging
    makes every KV head its own sequence, carrying its GQA group of query
    heads (see vllm/v1/attention/backends/ragged_layout.py). Decode uses
    a single reshape per Q/K/V/O against precomputed per-layer metadata;
    prefill / mixed batches copy token-major → member-major to handle varying
    sequence lengths.

    This body always executes eagerly: under piecewise compilation the
    wrapping custom op (``vllm::unified_attention_ragged``) is a
    splitting point, so it runs between captured CUDA-graph pieces. Inputs may
    therefore be padded to a CUDA-graph capture size — every path below slices
    Q/K/V to ``attn_metadata.num_actual_tokens`` first and writes only the
    first ``num_actual_tokens`` rows of ``output``. Padded output rows keep
    whatever ``torch.empty`` produced; the runner discards them, exactly as it
    does for the standard attention path.
    """
    assert layer.use_output, (
        "Ragged paging requires backends that accept an output "
        "buffer (FlashAttention does)."
    )

    # Compression scoring (Tangram): query/key scorers run here — inside the
    # eager splitting op, on the same unreshaped token-major tensors the
    # replaced module pre-hook used to see — because torch.compile skips
    # pre-hooks when inlining module forwards. Request offsets are unpadded,
    # so scorer slices never touch capture-size padding rows. No-op unless
    # the runner marked this step compression-active.
    if layer.compression_qk_scorer is not None:
        layer.compression_qk_scorer(query, key, value)

    num_groups = layer.num_groups_per_layer
    assert layer.page_group_size is not None
    head_size = layer.head_size
    num_heads = layer.num_heads
    num_kv_heads = layer.num_kv_heads
    # Column-major paging makes each KV head its own varlen sequence (one
    # member), so a member carries the GQA group's query heads.
    num_query_heads_per_kv = num_heads // num_kv_heads
    hidden = num_heads * head_size

    # Profiling and compilation dummy runs carry no metadata; zero the output
    # so downstream layers see finite values.
    if attn_metadata is None:
        output.fill_(0)
        return

    assert (getattr(attn_metadata, "num_head_groups_per_layer", 0)
            == num_groups), (
        "FlashAttentionMetadata.num_head_groups_per_layer "
        f"({getattr(attn_metadata, 'num_head_groups_per_layer', 0)}) "
        f"does not match layer's num_groups_per_layer ({num_groups}).")

    # Actual (unpadded) token count. Slicing before the member-major
    # reshapes both keeps the layout math correct under padding and bounds
    # the cost of the non-contiguous copies to real tokens.
    num_actual = attn_metadata.num_actual_tokens
    output_2d = output.view(-1, hidden)

    if attn_metadata.ragged_decode_layout:
        _ragged_decode_forward(
            layer, query, key, value, output_2d, kv_cache, attn_metadata,
            num_actual=num_actual, num_kv_heads=num_kv_heads,
            num_query_heads_per_kv=num_query_heads_per_kv, head_size=head_size,
        )
        return

    _ragged_member_major_forward(
        layer, query, key, value, output_2d, kv_cache, attn_metadata,
        num_actual=num_actual, num_heads=num_heads, num_kv_heads=num_kv_heads,
        num_query_heads_per_kv=num_query_heads_per_kv, head_size=head_size,
    )


def _ragged_decode_forward(
    layer: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_2d: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    *,
    num_actual: int,
    num_kv_heads: int,
    num_query_heads_per_kv: int,
    head_size: int,
) -> None:
    """Uniform-decode ragged path: one query token per (request, KV-head).

    Token-major Q/K/V flattens directly into member-major (one sequence per KV
    head) because every (req, member) sequence is a single row. Per-layer
    metadata is precomputed by the builder, so this avoids the per-layer
    view/slice CPU dispatch cost called out in ``FlashAttentionImpl.forward``.
    Writes into ``output_2d`` in place.

    ``reshape`` (not ``view``) because Q/K/V may be non-contiguous: models that
    fuse the QKV projection hand each tensor out as a ``split`` view, and a
    tensor with no post-projection op (e.g. Gemma-3's value, which unlike
    query/key is not re-materialized by q/k-norm + RoPE) keeps the fused row
    stride. Merging the token and ``num_kv_heads`` dims then spans that stride,
    which ``view`` rejects. ``reshape`` is a free view when already contiguous
    and copies only the non-contiguous case.
    """
    q_grouped = query[:num_actual].reshape(
        num_actual * num_kv_heads, num_query_heads_per_kv, head_size)
    k_grouped = key[:num_actual].reshape(
        num_actual * num_kv_heads, 1, head_size)
    v_grouped = value[:num_actual].reshape(
        num_actual * num_kv_heads, 1, head_size)
    output_grouped = output_2d[:num_actual].view(
        num_actual * num_kv_heads, num_query_heads_per_kv, head_size)
    layer_md = attn_metadata.per_layer_md[layer.layer_idx]
    layer.impl.forward(
        layer, q_grouped, k_grouped, v_grouped,
        kv_cache, layer_md, output=output_grouped,
    )


def _ragged_member_major_forward(
    layer: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_2d: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    *,
    num_actual: int,
    num_heads: int,
    num_kv_heads: int,
    num_query_heads_per_kv: int,
    head_size: int,
) -> None:
    """Member-major ragged path (prefill / mixed batches).

    A data copy to member-major is required because (req, member) sequences are
    not contiguous in any single view of token-major Q/K/V when sequence
    lengths vary. Builds this layer's virtual block table on demand, runs the
    attention, then writes the inverse (member-major -> token-major) reshape
    straight into ``output_2d`` in place.
    """
    q_token_major = query[:num_actual].view(-1, num_heads, head_size)
    k_token_major = key[:num_actual].view(-1, num_kv_heads, head_size)
    v_token_major = value[:num_actual].view(-1, num_kv_heads, head_size)

    cluster_block_table = attn_metadata.cluster_block_table
    clusters_per_layer = attn_metadata.clusters_per_layer
    cols_per_layer = attn_metadata.cols_per_layer
    seq_lens_grouped = attn_metadata.seq_lens_grouped
    slot_mapping_grouped = attn_metadata.slot_mapping_grouped
    query_start_loc_grouped = attn_metadata.query_start_loc_grouped
    assert cluster_block_table is not None
    assert clusters_per_layer is not None
    assert cols_per_layer is not None
    assert seq_lens_grouped is not None
    assert slot_mapping_grouped is not None
    assert query_start_loc_grouped is not None

    def to_member_major(
        x: torch.Tensor, sub_heads: int,
    ) -> torch.Tensor:
        # x: [num_actual, num_kv_heads * sub_heads, head_size] ->
        # [num_kv_heads * num_actual, sub_heads, head_size], one KV head
        # (member) per varlen sequence.
        return (
            x.view(num_actual, num_kv_heads, sub_heads, head_size)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(num_kv_heads * num_actual, sub_heads, head_size)
        )

    q_grouped = to_member_major(q_token_major, num_query_heads_per_kv)
    k_grouped = to_member_major(k_token_major, 1)
    v_grouped = to_member_major(v_token_major, 1)

    output_grouped = torch.empty(
        (num_kv_heads * num_actual, num_query_heads_per_kv, head_size),
        dtype=query.dtype,
        device=query.device,
    )

    # Slice the per-layer span of the member-major precomputes.
    layer_start = layer.layer_idx * num_kv_heads
    layer_end = layer_start + num_kv_heads

    num_reqs = seq_lens_grouped.shape[1]
    # This layer's virtual block table, built on demand: member-major
    # [num_kv_heads, num_reqs, max_blocks] -> [num_kv_heads * num_reqs,
    # max_blocks].
    block_table_layer = member_virtual_block_table(
        cluster_block_table, clusters_per_layer[layer.layer_idx],
        cols_per_layer[layer.layer_idx], attn_metadata.page_group_size,
        cluster_axis=1,
    ).permute(1, 0, 2).reshape(num_kv_heads * num_reqs, -1)
    seq_lens_layer = seq_lens_grouped[layer_start:layer_end].reshape(
        num_kv_heads * num_reqs)
    slot_mapping_layer = slot_mapping_grouped[
        layer_start:layer_end].reshape(-1)

    layer_md = layer_overlay(
        attn_metadata,
        num_actual_tokens=num_kv_heads * num_actual,
        block_table=block_table_layer,
        seq_lens=seq_lens_layer,
        slot_mapping=slot_mapping_layer,
        query_start_loc=query_start_loc_grouped,
    )

    layer.impl.forward(
        layer, q_grouped, k_grouped, v_grouped,
        kv_cache, layer_md, output=output_grouped,
    )

    # Inverse reshape member-major → token-major, written straight into the
    # caller's output buffer. The ``copy_`` is the single materializing copy
    # on this path, replacing the ``.contiguous()`` the pre-custom-op code
    # used before returning a fresh tensor.
    output_2d[:num_actual].view(
        num_actual, num_kv_heads, num_query_heads_per_kv, head_size,
    ).copy_(
        output_grouped.view(
            num_kv_heads, num_actual, num_query_heads_per_kv, head_size,
        ).permute(1, 0, 2, 3)
    )
