# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request KV cache reordering after compression.

Runs after ``model.forward``. For every (layer, head-group) it turns the
caches ``prepare_keep_decision`` stashed into one :class:`EvictionPlan` --
``keep = sink ∪ locked ∪ topk ∪ tail`` per cluster -- and hands the plan to a
:class:`KeptKVWriteback`, which moves the kept K/V into block-aligned form.
Touches only KV tensors and the block_table."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.attention.compression.compressor import KVCompressor
from vllm.v1.attention.compression.eviction_writeback import (
    NUM_PLAN_COLS,
    EvictionPlan,
    KeptKVWriteback,
    PlanCol,
    TorchWriteback,
)
from vllm.v1.worker.block_table import BlockTable


def _new_region_per_group(
    kept: np.ndarray,
    sink_size: int,
    locked: np.ndarray,
    tail_size: int,
    eval_len: np.ndarray | None = None,
) -> np.ndarray:
    """Recover the block-aligned top-k ("new") span from ``kept``, per group.

    ``kept`` packs ``sink + locked + new + tail`` and the writeback needs
    ``new`` back. ``eval_len`` clamps it to the scored region when the caller
    slices ``sorted_idx``; the keep-all path takes no slice and omits it.
    """
    new = np.maximum(0, kept - sink_size - locked - tail_size)
    if eval_len is not None:
        new = np.minimum(new, eval_len)
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
        writeback: KeptKVWriteback | None = None,
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
        # Moves the kept KV; the torch reference unless the runner picks the
        # batched kernel.
        self.writeback = (
            writeback if writeback is not None else TorchWriteback(block_size))

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
        metadata = compression_metadata

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

        block_table_gpu = block_table.block_table.gpu
        bt_row_offset = metadata.row_idx * block_table_gpu.stride(0)
        bt_entry_stride = block_table_gpu.stride(1)
        new_locked_all = np.zeros((num_compressed, num_groups), dtype=np.int64)
        chunk_len = metadata.chunk_len
        plan_rows: list[np.ndarray] = []

        for static_idx, layer_idx in enumerate(compressed_layer_ids):
            prev_seq_lens_np = prev_seq_lens_static_cpu[static_idx].astype(
                np.int64, copy=True)
            total_seen = prev_seq_lens_np + chunk_len
            if int(total_seen.max()) == 0:
                raise RuntimeError(
                    f"CompressionExecutor.run_request(layer={layer_idx}"
                    f", req={metadata.req_id}): total_seen=0 (prev=0, "
                    f"chunk_len={chunk_len}). Skip the compression step "
                    "instead of calling run_request.")
            locked = locked_cpu[static_idx].astype(np.int64)
            kept = kept_lengths_all[static_idx].astype(np.int64)

            # Fast path: keep everything -> only refresh new_locked.
            if adjusted_ratio >= 1.0:
                new_locked_all[static_idx] = locked + _new_region_per_group(
                    kept, sink_size, locked, tail_size)
                continue

            # Under TP this rank may extend its top-k to the MAX-reduced
            # length; sorted_idx holds eval_len, so the slice is safe.
            k_aligned = _new_region_per_group(
                kept, sink_size, locked, tail_size, real_eval_len[static_idx])
            new_locked_all[static_idx] = locked + k_aligned

            groups = np.flatnonzero(kept > 0)
            if groups.size == 0:
                continue
            rows = np.empty((groups.size, NUM_PLAN_COLS), dtype=np.int64)
            rows[:, PlanCol.LAYER] = layer_idx
            rows[:, PlanCol.BT_OFFSET] = bt_row_offset + (
                layer_idx * num_groups + groups) * bt_entry_stride
            rows[:, PlanCol.CLUSTER] = static_idx * num_groups + groups
            rows[:, PlanCol.KEPT_LO] = sink_size + locked[groups]
            rows[:, PlanCol.K_ALIGNED] = k_aligned[groups]
            rows[:, PlanCol.TAIL_LO] = total_seen[groups] - tail_size
            rows[:, PlanCol.KEPT] = kept[groups]
            plan_rows.append(rows)

        if plan_rows:
            plan = EvictionPlan(
                table=np.concatenate(plan_rows),
                sink_size=sink_size,
                tail_size=tail_size,
                eval_len=eval_len,
            )
            # The score store follows the very same positions, so a survivor's
            # statistics move with it and an evicted one's are released.
            self.writeback.run(
                plan,
                layer_kv_caches,
                block_table_gpu,
                compressor.workspace.sorted_index.view(
                    num_compressed * num_groups, self.page_group_size, -1),
                compressor.compaction_target(metadata.req_id),
            )

        # The compressor owns this state -- it lives in the preallocated
        # workspace -- so the executor reports rather than writes it.
        compressor.commit_chunk(
            metadata.req_id, new_locked_all, kept_lengths_all)
        req.borrowed_sorted_indices = None

        return kept_lengths_all
