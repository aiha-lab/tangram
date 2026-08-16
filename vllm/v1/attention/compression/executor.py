# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request KV cache reordering after compression.

Runs after ``model.forward``. For each (layer, head-group) it gathers
the cached KV, assembles ``keep_idx = sink ∪ locked ∪ topk ∪ tail``
from the caches ``prepare_keep_decision`` stashed, and scatters the
kept K/V back in block-aligned form. Backend-agnostic: touches only
KV tensors and the block_table."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.attention.compression.compressor import KVCompressor
from vllm.v1.worker.block_table import BlockTable


def _new_region_from_kept_length(
    kept_length: int,
    sink_size: int,
    locked: int,
    tail_size: int,
    eval_len: int | None = None,
) -> int:
    """Recover the block-aligned top-k ("new") span from ``kept_length``.

    ``kept_length`` is produced once by
    ``KVCompressor.compute_kept_lengths_per_rank`` and packs
    ``sink + locked + new + tail`` slots; the writeback needs the ``new``
    region back out. ``eval_len`` clamps it to the scored region when the
    caller slices ``sorted_idx``; the keep-all fast path takes no slice and
    omits the clamp.
    """
    new = max(0, kept_length - sink_size - locked - tail_size)
    if eval_len is not None:
        new = min(new, eval_len)
    return new


@dataclass
class CompressionMetadata:
    """Per-(request, step) compression info passed to ``run_request``.

    ``floor_min`` is the per-(layer, group) ``kept_lengths`` absolute
    floor; 0 disables it. All other run-shape config lives on the
    compressor under ``req_state[req_id]``.
    """
    req_id: str
    row_idx: int
    chunk_len: int
    floor_min: int


class CompressionExecutor:
    """One instance per ModelRunner; stateless across requests."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads_per_layer: int,
        page_group_size: int,
        head_size: int,
        block_size: int,
        compressed_layer_ids: list[int] | None = None,
    ) -> None:
        assert num_kv_heads_per_layer % page_group_size == 0, (
            f"num_kv_heads_per_layer ({num_kv_heads_per_layer}) must be "
            f"divisible by page_group_size ({page_group_size})."
        )
        self.num_layers = num_layers
        self.num_kv_heads_per_layer = num_kv_heads_per_layer
        self.page_group_size = page_group_size
        self.num_head_groups_per_layer = (
            num_kv_heads_per_layer // page_group_size
        )
        self.head_size = head_size
        self.block_size = block_size
        # Physical indices of the compressible (full-attention) layers, in the
        # same order as the compressor's per-layer state. For a dense model
        # every layer is compressible, so this is ``range(num_layers)`` and the
        # loop below is unchanged. For sliding-window hybrids (e.g. gemma-3,
        # gpt-oss) only the full-attention layers appear here; sliding-window
        # layers keep their full KV and are skipped (never evicted). The
        # compressor's caches are indexed by the *compressed* position
        # ``static_idx`` while KV / block-table access uses the *physical*
        # layer ``compressed_layer_ids[static_idx]``.
        if compressed_layer_ids is None:
            compressed_layer_ids = list(range(num_layers))
        self.compressed_layer_ids = compressed_layer_ids
        self.num_compressed_layers = len(compressed_layer_ids)
        # Reused arange slabs; sink/win sizes are KeepDecision-uniform.
        self._sink_idx_cache: torch.Tensor | None = None
        self._tail_idx_cache: torch.Tensor | None = None

    def run_request(
        self,
        layer_kv_caches: list[torch.Tensor],
        block_table: BlockTable,
        prev_seq_lens_static_cpu: np.ndarray,
        compressor: KVCompressor,
        compression_metadata: CompressionMetadata,
    ) -> np.ndarray:
        """Apply the keep decision to every compressible layer in one call.

        Compressible layers are the full-attention layers (all layers for a
        dense model; only the non-sliding layers for a sliding-window hybrid).
        Slots ``[0, kept_lengths[compressed_layer, group])`` of each cache are
        overwritten with the kept KV (block-aligned write). The
        ``block_table`` is not mutated; the caller invokes
        ``compact_after_compress_all_layers`` afterwards. Returns
        ``[num_compressed, num_groups]`` int32 post-evict lengths, indexed by
        compressed position (not physical layer).

        ``prev_seq_lens_static_cpu`` is the ``[num_compressed, num_groups]``
        per-compressible-layer valid length as of the previous compression
        boundary (the kept_lengths the last eviction left). It is passed
        explicitly rather than read from the worker's ``effective_seq_lens``
        because, once a chunk is sliced across forward steps, that array holds
        the raw write extent (kept + accumulated), not the pre-chunk length the
        keep arithmetic needs. ``total_seen = prev + chunk_len`` still equals
        the physical written extent, so the block reads are unchanged.

        Requires ``prepare_keep_decision`` to have populated
        ``req.cross_layer_decision`` and the per-(layer, group) caches.
        """
        assert block_table.ragged, (
            "CompressionExecutor.run_request requires a ragged "
            "BlockTable."
        )
        assert len(layer_kv_caches) == self.num_layers

        # ``num_compressed`` sizes the compressor's per-compressed-layer state
        # (kept_lengths, locked, sorted_idx); ``compressed_layer_ids`` maps a
        # compressed position to its physical layer for KV / block-table access.
        num_compressed = self.num_compressed_layers
        compressed_layer_ids = self.compressed_layer_ids
        num_groups = self.num_head_groups_per_layer
        block_size = self.block_size
        metadata = compression_metadata
        device = layer_kv_caches[0].device

        req = compressor.req_state.get(metadata.req_id)
        if req is None or req.cross_layer_decision is None:
            raise RuntimeError(
                f"CompressionExecutor.run_request({metadata.req_id}): "
                "cross_layer_decision missing — prepare_keep_decision "
                "must run before run_request.")
        keep_dec = req.cross_layer_decision
        sink_size = keep_dec.sink_size
        # Trailing always-kept span: the recent window under the ratio regime,
        # the whole fresh chunk under the budget regime (unless configured
        # otherwise). The writeback only needs its width.
        tail_size = keep_dec.tail_size
        adjusted_ratio = keep_dec.adjusted_ratio
        eval_len = keep_dec.eval_len
        # Genuine (unpadded) eval width per (layer, group): uniform under the
        # ratio regime, ragged under the budget regime, where the score tensor
        # is padded to the widest group. Slicing ``sorted_idx`` past a group's
        # own width would pick padding positions.
        real_eval_len = (
            req.real_eval_len_cpu
            if req.real_eval_len_cpu is not None
            else np.full((num_compressed, num_groups), eval_len,
                         dtype=np.int64))

        # Under TP the runner cross-rank MAX-reduces kept_lengths before
        # reaching us.
        if req.cached_kept_lengths_cpu is None:
            raise RuntimeError(
                f"CompressionExecutor.run_request({metadata.req_id}): "
                "cached_kept_lengths_cpu missing — "
                "compute_kept_lengths_per_rank must run before "
                "run_request.")
        kept_lengths_all = req.cached_kept_lengths_cpu

        locked_cpu = (
            req.locked_count_cpu
            if req.locked_count_cpu is not None
            else np.zeros((num_compressed, num_groups), dtype=np.int64))
        sorted_idx = req.cached_sorted_indices

        if (self._sink_idx_cache is None
                or self._sink_idx_cache.numel() < sink_size
                or self._sink_idx_cache.device != device):
            self._sink_idx_cache = torch.arange(
                max(sink_size, 64), device=device, dtype=torch.long)
        sink_idx_full = self._sink_idx_cache[:sink_size]
        if (self._tail_idx_cache is None
                or self._tail_idx_cache.numel() < tail_size
                or self._tail_idx_cache.device != device):
            self._tail_idx_cache = torch.arange(
                max(tail_size, 4096), device=device, dtype=torch.long)
        tail_idx_base = self._tail_idx_cache[:tail_size]

        block_table_gpu = block_table.block_table.gpu
        new_locked_all = np.zeros((num_compressed, num_groups), dtype=np.int64)
        chunk_len = metadata.chunk_len
        row_idx = metadata.row_idx

        for static_idx, layer_idx in enumerate(compressed_layer_ids):
            # ``layer_idx`` is the physical layer (KV cache + block-table);
            # ``static_idx`` indexes the compressor's per-compressed-layer
            # caches (kept_lengths, locked, sorted_idx, layer_states). For a
            # dense model the two coincide.
            kv_cache = layer_kv_caches[layer_idx]
            layer_first = layer_idx * num_groups

            prev_seq_lens_np = prev_seq_lens_static_cpu[static_idx].astype(
                np.int64, copy=True)
            total_seen_per_group = prev_seq_lens_np + chunk_len
            total_seen_max = int(total_seen_per_group.max())
            if total_seen_max == 0:
                raise RuntimeError(
                    f"CompressionExecutor.run_request(layer={layer_idx}"
                    f", req={metadata.req_id}): total_seen=0 (prev=0, "
                    f"chunk_len={chunk_len}). Skip the compression step "
                    "instead of calling run_request.")

            # Fast path: keep everything → only refresh new_locked.
            if adjusted_ratio >= 1.0:
                for group_idx in range(num_groups):
                    locked = int(locked_cpu[static_idx, group_idx])
                    kept_length = int(
                        kept_lengths_all[static_idx, group_idx])
                    k_aligned = _new_region_from_kept_length(
                        kept_length, sink_size, locked, tail_size)
                    new_locked_all[static_idx, group_idx] = (
                        locked + k_aligned)
                continue

            for group_idx in range(num_groups):
                total_seen = int(total_seen_per_group[group_idx])
                locked = int(locked_cpu[static_idx, group_idx])
                kept_lo = sink_size + locked
                tail_lo = total_seen - tail_size

                # Under TP this rank may need to extend its top-k up to
                # the cross-rank-MAX-reduced kept_length; sorted_idx
                # already holds eval_len positions so the slice is safe.
                kept_length = int(kept_lengths_all[static_idx, group_idx])
                k_aligned = _new_region_from_kept_length(
                    kept_length, sink_size, locked, tail_size,
                    int(real_eval_len[static_idx, group_idx]))
                new_locked_all[static_idx, group_idx] = locked + k_aligned

                if kept_length == 0:
                    continue

                n_blocks = (total_seen + block_size - 1) // block_size
                block_ids = block_table_gpu[
                    row_idx, layer_first + group_idx, :n_blocks
                ].long()
                keep_positions = self._gather_and_writeback_kept_kv(
                    kv_cache=kv_cache,
                    block_ids=block_ids,
                    sink_idx=sink_idx_full,
                    locked=locked,
                    k_aligned=k_aligned,
                    kept_lo=kept_lo,
                    sorted_idx_group=(
                        sorted_idx[static_idx, group_idx]
                        if sorted_idx is not None else None),
                    tail_idx=tail_idx_base,
                    tail_lo=tail_lo,
                    kept_length=kept_length,
                    device=device,
                )
                # Keep the regime's score memory slot-aligned with the KV it
                # just rewrote, using the very same positions: an evicted
                # position's statistics are released, a surviving position's
                # follow it to its new slot. No-op under the ratio regime.
                compressor.compact_cluster_stats(
                    metadata.req_id, static_idx, group_idx,
                    keep_positions, kept_length)

        # Batched H2D: collapses L per-layer state writes into two. State is
        # indexed by compressed position (matches the compressor's caches).
        new_locked_gpu = torch.from_numpy(new_locked_all).to(device)
        valid_lengths_gpu = torch.from_numpy(
            kept_lengths_all.astype(np.int64)).to(device)
        for static_idx in range(num_compressed):
            state = req.layer_states.get(static_idx)
            if state is None:
                continue
            state.locked_count_per_group = new_locked_gpu[static_idx]
            state.valid_lengths_per_group = valid_lengths_gpu[static_idx]

        return kept_lengths_all

    def _gather_and_writeback_kept_kv(
        self,
        kv_cache: torch.Tensor,
        block_ids: torch.Tensor,
        sink_idx: torch.Tensor,
        locked: int,
        k_aligned: int,
        kept_lo: int,
        sorted_idx_group: torch.Tensor | None,
        tail_idx: torch.Tensor,
        tail_lo: int,
        kept_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Evict one (layer, group): gather the kept KV positions, write back.

        ``block_ids`` are the cluster's physical blocks covering the
        pre-eviction span. Its column-major page
        ``[2, n_blocks, page_group_size, block_size, head_size]`` is viewed
        token-major (a free permute, no copy) so the kept positions can be
        gathered directly — the full slab is never materialised, avoiding the
        O(total_seen) copy a ``.reshape`` of the permuted tensor would force.
        The kept KV is then written back block-aligned into the same blocks,
        with the trailing partial block zero-padded.

        The kept positions form a ``[page_group_size, kept_length]`` matrix:
        every column (KV head) keeps the same sink / locked / tail positions,
        and only the middle ``k_aligned`` block — that column's own top-k by
        score, taken from ``sorted_idx_group`` (the per-(layer, group)
        descending ranking, shape ``[page_group_size, eval_len]``) — differs.
        ``sorted_idx_group`` is dereferenced only when ``k_aligned > 0``; the
        fast / zero paths pass ``k_aligned == 0`` (and may pass ``None``). The
        shared sink/locked/k_aligned/tail counts make every column the same
        length, so the pad + write-back are column-uniform.

        Returns that matrix, so the caller can apply the identical positions to
        anything else stored per cache slot (the regime's score memory).

        See vllm/v1/attention/backends/ragged_layout.py for the layout.
        """
        page_group_size = self.page_group_size
        block_size = self.block_size
        head_size = self.head_size
        sink_size = int(sink_idx.numel())

        # Token-major view of the cluster's blocks: permuting block_size ahead
        # of page_group_size keeps it a view, so token t lives at block
        # t // block_size, offset t % block_size.
        slab_view = kv_cache[:, block_ids].permute(0, 1, 3, 2, 4)

        col_parts: list[torch.Tensor] = []
        if sink_size > 0:
            col_parts.append(
                sink_idx.unsqueeze(0).expand(page_group_size, -1))
        if locked > 0:
            col_parts.append(
                torch.arange(
                    sink_size, sink_size + locked,
                    device=device, dtype=torch.long)
                .unsqueeze(0).expand(page_group_size, -1))
        if k_aligned > 0:
            mid = sorted_idx_group[:, :k_aligned]
            mid, _ = mid.sort(dim=-1)
            col_parts.append(mid + kept_lo)
        if tail_idx.numel() > 0:
            col_parts.append(
                (tail_idx + tail_lo).unsqueeze(0).expand(page_group_size, -1))
        keep_mat = (
            torch.cat(col_parts, dim=1) if col_parts
            else torch.empty(
                page_group_size, 0, dtype=torch.long, device=device))

        keep_block = torch.div(keep_mat, block_size, rounding_mode="floor")
        keep_offset = keep_mat - keep_block * block_size
        col_ix = torch.arange(
            page_group_size, device=device, dtype=torch.long
        ).unsqueeze(1).expand_as(keep_mat)
        # Advanced-index (block, offset, column) -> [2, page_group_size,
        # kept_length, head_size].
        kept_kv = slab_view[
            :, keep_block, keep_offset, col_ix
        ].permute(0, 2, 1, 3).contiguous()

        # Write back block-aligned; zero-pad the trailing partial block.
        n_blocks_write = (kept_length + block_size - 1) // block_size
        padded_size = n_blocks_write * block_size
        if kept_length < padded_size:
            pad = torch.zeros(
                2, padded_size - kept_length, page_group_size, head_size,
                dtype=kept_kv.dtype, device=device)
            kept_kv = torch.cat([kept_kv, pad], dim=1)
        # Inverse of the gather: token-major slab back to column-major
        # [2, n_blocks_write, page_group_size, block_size, head_size].
        kv_cache[:, block_ids[:n_blocks_write]] = (
            kept_kv.view(
                2, n_blocks_write, block_size, page_group_size, head_size)
            .permute(0, 1, 3, 2, 4)
        )
        return keep_mat
