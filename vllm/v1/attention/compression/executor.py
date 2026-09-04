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

    ``kept_length`` packs ``sink + locked + new + tail`` and the writeback
    needs ``new`` back. ``eval_len`` clamps it to the scored region when the
    caller slices ``sorted_idx``; the keep-all path takes no slice and omits it.
    """
    new = max(0, kept_length - sink_size - locked - tail_size)
    if eval_len is not None:
        new = min(new, eval_len)
    return new


@dataclass
class CompressionMetadata:
    """Per-(request, step) compression info passed to ``run_request``.

    ``floor_min`` is the per-entry absolute ``kept_lengths`` floor, 0 to
    disable. Everything else lives on the compressor's ``req_state``.
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
        # The compressor's caches are indexed by COMPRESSED position while KV
        # and block-table access uses the PHYSICAL layer; for a dense model
        # the two coincide.
        if compressed_layer_ids is None:
            compressed_layer_ids = list(range(num_layers))
        self.compressed_layer_ids = compressed_layer_ids
        self.num_compressed_layers = len(compressed_layer_ids)
        # Reused arange slabs; sink/tail sizes are KeepDecision-uniform.
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

        Slots ``[0, kept_lengths[entry])`` are overwritten with the kept KV,
        block-aligned, and ``block_table`` is NOT mutated -- the caller runs
        ``compact_after_compress_all_layers`` afterwards. Returns
        ``[num_compressed, num_groups]`` int32 by compressed position, and
        requires ``prepare_keep_decision`` to have run.

        ``prev_seq_lens_static_cpu`` is the length as of the previous boundary,
        passed explicitly rather than read from ``effective_seq_lens`` because
        once a chunk is sliced across steps that array holds the raw write
        extent instead. ``total_seen = prev + chunk_len`` still equals the
        written extent, so the block reads are unchanged.
        """
        assert block_table.ragged, (
            "CompressionExecutor.run_request requires a ragged "
            "BlockTable."
        )
        assert len(layer_kv_caches) == self.num_layers

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
        # The writeback needs only the always-kept tail's width.
        tail_size = keep_dec.tail_size
        adjusted_ratio = keep_dec.adjusted_ratio
        eval_len = keep_dec.eval_len
        # Ragged under the budget regime, where the score tensor is padded to
        # the widest group: slicing ``sorted_idx`` past a group's own width
        # would pick padding.
        real_eval_len = req.real_eval_len_cpu

        # Under TP the runner cross-rank MAX-reduces kept_lengths before
        # reaching us.
        if req.cached_kept_lengths_cpu is None:
            raise RuntimeError(
                f"CompressionExecutor.run_request({metadata.req_id}): "
                "cached_kept_lengths_cpu missing — "
                "compute_kept_lengths_per_rank must run before "
                "run_request.")
        kept_lengths_all = req.cached_kept_lengths_cpu

        locked_cpu = req.locked_count_cpu
        sorted_idx = req.borrowed_sorted_indices

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

                # Under TP this rank may extend its top-k to the MAX-reduced
                # length; sorted_idx holds eval_len, so the slice is safe.
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
                # The very same positions, so a survivor's statistics follow
                # it to its new slot and an evicted one's are released.
                compressor.compact_cluster_stats(
                    metadata.req_id, static_idx, group_idx,
                    keep_positions, kept_length)

        # The compressor owns this state -- it lives in the preallocated
        # workspace -- so the executor reports rather than writes it.
        compressor.commit_chunk(
            metadata.req_id, new_locked_all, kept_lengths_all)
        req.borrowed_sorted_indices = None

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

        The gather indexes the pages directly, so it costs O(kept) and never
        builds the cluster's KV as one slab -- an intermediate that would cost
        O(total_seen) however little survives. The result is written back
        block-aligned into the same blocks, trailing partial block zero-padded.

        The kept positions form a ``[page_group_size, kept_length]`` matrix in
        which every column keeps the SAME sink / locked / tail and only the
        middle ``k_aligned`` span differs, so the pad and write-back are
        column-uniform. It is returned, so the caller can apply the identical
        positions to anything else stored per cache slot.
        ``sorted_idx_group`` is dereferenced only when ``k_aligned > 0``.
        """
        page_group_size = self.page_group_size
        block_size = self.block_size
        head_size = self.head_size
        sink_size = int(sink_idx.numel())

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
        # Straight from the pages: ``block_ids[keep_block]`` picks the page,
        # ``col_ix`` the column, ``keep_offset`` the token within the block.
        # Going through a token-major view of the cluster instead would read
        # every page it holds, kept or not -- that view is an advanced index,
        # so it copies -- and this reads only what survives.
        # -> [2, page_group_size, kept, head_size].
        kept_kv = kv_cache[
            :, block_ids[keep_block], col_ix, keep_offset
        ].permute(0, 2, 1, 3).contiguous()

        # Write back block-aligned; zero-pad the trailing partial block.
        n_blocks_write = (kept_length + block_size - 1) // block_size
        padded_size = n_blocks_write * block_size
        if kept_length < padded_size:
            pad = torch.zeros(
                2, padded_size - kept_length, page_group_size, head_size,
                dtype=kept_kv.dtype, device=device)
            kept_kv = torch.cat([kept_kv, pad], dim=1)
        # Inverse of the gather: token-major slab back to column-major.
        kv_cache[:, block_ids[:n_blocks_write]] = (
            kept_kv.view(
                2, n_blocks_write, block_size, page_group_size, head_size)
            .permute(0, 1, 3, 2, 4)
        )
        return keep_mat
