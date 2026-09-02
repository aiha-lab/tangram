# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The block table for ragged paging: one page per (head group, block).

Ragged paging gives every head group its own pages, so this block table
carries a group axis the base one does not: ``[request, head group, slot]``
instead of ``[request, slot]``, and a slot mapping of ``[head group, token]``
instead of ``[token]``. The group index is flat over layers -- group
``layer * num_head_groups_per_layer + g`` -- so the layer dimension needs no
separate axis and no per-layer tiling.

What the group axis buys is the whole point of compression: each group evicts
independently, so a request's groups sit at *different* depths. Two operations
exist only because of that, and they trim from opposite ends:

    compact_after_compress_all_layers  keeps the front, frees the tail
    null_front_blocks_sliding          keeps the tail, nulls the front

The second leaves ``num_blocks_per_row`` untouched on purpose. A sliding-window
layer's surviving blocks must not change column, or every position-to-block
mapping shifts and the output changes; only the pages themselves go back to the
pool.

Non-uniform depths are also why ``snapshot_row`` / ``restore_row`` exist:
``add_row`` rebuilds a row by filling every group to one uniform depth, which
cannot express a compressed row, so a request dropped from the persistent batch
has its row captured verbatim and written back on re-add.
"""
import numpy as np
import torch

from vllm.v1.worker.block_table import BlockTable


class RaggedBlockTable(BlockTable):
    """``BlockTable`` with a head-group axis on every buffer."""

    ragged = True

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
        num_head_groups: int,
        num_head_groups_per_layer: int | None = None,
    ):
        """
        Args:
            num_head_groups: Total head groups
                (``num_head_groups_per_layer x num_layers``) -- the group axis
                of every buffer.
            num_head_groups_per_layer: Per-layer head-group count, for
                layer-aware consumers. Defaults to ``num_head_groups`` so
                single-layer unit tests can omit it; production always passes
                the real value.

        Every other argument is the base class's.
        """
        assert num_head_groups > 0, "num_head_groups must be a positive int."
        assert kernel_block_size == block_size, (
            "Ragged paging requires kernel_block_size == "
            "block_size (hybrid kernel block sizes not supported)."
        )
        if num_head_groups_per_layer is None:
            num_head_groups_per_layer = num_head_groups
        assert num_head_groups % num_head_groups_per_layer == 0, (
            "num_head_groups_per_layer must divide num_head_groups; "
            f"got num_head_groups={num_head_groups}, "
            f"num_head_groups_per_layer={num_head_groups_per_layer}.")

        # Read by the base constructor's buffer allocation, so set first.
        self.num_head_groups = num_head_groups
        self.num_head_groups_per_layer = num_head_groups_per_layer

        super().__init__(
            block_size,
            max_num_reqs,
            max_num_blocks_per_req,
            max_num_batched_tokens,
            pin_memory,
            device,
            kernel_block_size,
            cp_kv_cache_interleave_size,
        )

        assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
            "Ragged paging is not implemented for DCP/PCP world sizes > 1."
        )

    def _allocate_buffers(self) -> None:
        self.block_table = self._make_buffer(
            self.max_num_reqs,
            self.num_head_groups,
            self.max_num_blocks_per_req,
            dtype=torch.int32,
        )
        self.num_blocks_per_row = np.zeros(
            (self.max_num_reqs, self.num_head_groups), dtype=np.int32
        )
        self.slot_mapping = self._make_buffer(
            self.num_head_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
        )

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        if not block_ids:
            return
        self._append_row_grouped(block_ids, row_idx)

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx, :] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        block_table_np = self.block_table.np
        block_table_np[tgt, :, :] = block_table_np[src, :, :]
        self.num_blocks_per_row[tgt, :] = self.num_blocks_per_row[src, :]

    def swap_row(self, src: int, tgt: int) -> None:
        block_table_np = self.block_table.np
        tmp = block_table_np[src, :, :].copy()
        block_table_np[src, :, :] = block_table_np[tgt, :, :]
        block_table_np[tgt, :, :] = tmp
        tmp_n = self.num_blocks_per_row[src, :].copy()
        self.num_blocks_per_row[src, :] = self.num_blocks_per_row[tgt, :]
        self.num_blocks_per_row[tgt, :] = tmp_n

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        self._compute_slot_mapping_grouped(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.gpu[:num_reqs].copy_(
            self.block_table.cpu[:num_reqs], non_blocking=True
        )

    def commit_slot_mapping(self, num_tokens: int) -> None:
        self.slot_mapping.gpu[:, :num_tokens].copy_(
            self.slot_mapping.cpu[:, :num_tokens], non_blocking=True
        )

    def _append_row_grouped(self, block_ids: list[int], row_idx: int) -> None:
        """Ragged append.

        ``block_ids`` is a flat sequence sized by
        ``num_required_blocks × num_head_groups − len(req_blocks)``. With
        pre-append per-group counts ``starts[g]``, the target is a
        uniform per-group depth ``num_required = (sum(starts) + total) /
        num_head_groups``. We fill groups in order — group 0 takes its
        ``num_required − starts[0]`` ids first, then group 1, etc. Block
        ids are interchangeable, so the per-group order is purely an
        accounting choice; only the uniqueness of (row, group, slot) →
        id matters.
        """
        assert self.num_head_groups is not None
        num_groups = self.num_head_groups
        total = len(block_ids)

        starts = self.num_blocks_per_row[row_idx]
        sum_starts = int(starts.sum())
        # ``num_required`` must be an integer; otherwise allocator and
        # append are out of sync (e.g. stale ``num_required_blocks``).
        if (sum_starts + total) % num_groups != 0:
            raise RuntimeError(
                f"ragged append_row: total {total} + existing "
                f"{sum_starts} not divisible by num_head_groups "
                f"({num_groups}); uniform per-group target unrecoverable.")
        num_required = (sum_starts + total) // num_groups
        if num_required < int(starts.max()):
            raise RuntimeError(
                f"ragged append_row: num_required {num_required} < "
                f"max(starts) {int(starts.max())}; cannot shrink groups "
                "via append.")

        block_ids_np = np.asarray(block_ids, dtype=np.int32)
        pos = 0
        for group_idx in range(num_groups):
            start = int(starts[group_idx])
            num_new = num_required - start
            if num_new <= 0:
                continue
            self.block_table.np[
                row_idx, group_idx, start : start + num_new
            ] = block_ids_np[pos : pos + num_new]
            pos += num_new
        if pos != total:
            raise RuntimeError(
                f"ragged append_row: consumed {pos} block ids but received "
                f"{total}; per-group block counts are inconsistent and the "
                "KV block table would be corrupted.")
        self.num_blocks_per_row[row_idx, :] = num_required

    def _compute_slot_mapping_grouped(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        """Per-(group, token) slot mapping for the ragged 3D path.

        ``positions`` is either 1D ``[num_tokens]`` (every group shares the
        same position — pre-compression) or 2D
        ``[num_head_groups, num_tokens]`` (post-compression).
        """
        assert self.num_head_groups is not None
        num_groups = self.num_head_groups
        if positions.ndim == 1:
            num_tokens = positions.shape[0]
            positions = np.broadcast_to(
                positions[None, :], (num_groups, num_tokens))
        else:
            assert (positions.ndim == 2
                    and positions.shape[0] == num_groups), (
                f"positions must have shape [num_head_groups({num_groups}), "
                f"T]; got {positions.shape}.")
            num_tokens = positions.shape[1]

        # ``block_table.np`` shape ``[max_num_reqs, num_head_groups,
        # max_num_blocks_per_req]``; flatten the trailing dims and index
        # in one fancy-index pass.
        block_table_np = self.block_table.np
        max_blocks = self.max_num_blocks_per_req

        block_idx = positions // self.block_size
        group_axis = np.arange(num_groups, dtype=np.int64)[:, None]
        flat_indices = (
            req_indices.astype(np.int64)[None, :] * (num_groups * max_blocks)
            + group_axis * max_blocks
            + block_idx.astype(np.int64)
        )
        block_numbers = block_table_np.ravel()[flat_indices]
        block_offsets = positions % self.block_size
        slot = (
            block_numbers.astype(np.int64) * self.block_size + block_offsets)
        self.slot_mapping.np[:, :num_tokens] = slot

    def snapshot_row(self, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Capture a row's exact block ids + per-group fill counts.

        Ragged paging only. A compressed request's per-group block
        counts are non-uniform (each cluster evicts independently), and the
        flat ``CachedRequestState.block_ids`` list cannot reconstruct that
        layout via ``add_row`` (which fills groups uniformly, group-major).
        When such a request is dropped from the persistent batch (it skipped
        a step) we snapshot its live row here and ``restore_row`` it on
        re-add, preserving the exact (cluster, slot) -> block id mapping the
        KV already lives at."""
        return (
            self.block_table.np[row_idx, :, :].copy(),
            self.num_blocks_per_row[row_idx, :].copy(),
        )

    def restore_row(
        self, row_idx: int, snapshot: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Write back a row captured by :meth:`snapshot_row`."""
        block_ids_np, counts = snapshot
        self.block_table.np[row_idx, :, :] = block_ids_np
        self.num_blocks_per_row[row_idx, :] = counts

    def compact_after_compress_all_layers(
        self,
        row_idx: int,
        num_head_groups_per_layer: int,
        new_num_blocks_per_layer: np.ndarray,
    ) -> np.ndarray:
        """Trim trailing block ids for every layer's groups in ``row_idx``.

        For each (layer, group), block_table entries in
        ``[new_num_blocks[g], old_num_blocks[g])`` are zeroed and their physical
        ids returned (caller releases them via ``free_blocks_by_ids``).
        ``new_num_blocks_per_layer`` has shape ``[num_layers, groups]``.
        """
        num_groups = num_head_groups_per_layer
        new_counts = np.asarray(new_num_blocks_per_layer, dtype=np.int32)
        if new_counts.ndim != 2 or new_counts.shape[1] != num_groups:
            raise ValueError(
                f"new_num_blocks_per_layer shape {new_counts.shape} != "
                f"(num_layers, {num_groups}).")
        num_layers = new_counts.shape[0]
        total_groups = num_layers * num_groups

        old_2d = self.num_blocks_per_row[row_idx, :total_groups].reshape(
            num_layers, num_groups)
        if (new_counts > old_2d).any():
            raise RuntimeError(
                "compact_after_compress_all_layers cannot grow num_blocks; "
                f"max old={old_2d.max(axis=0).tolist()} "
                f"max new={new_counts.max(axis=0).tolist()}.")

        block_table_np = self.block_table.np
        # Fancy-index over all (layer, group) cells: each flat row
        # ``layer * num_groups + group`` has columns
        # ``[new_n, old_n)`` freed; equal counts contribute nothing.
        row_view = block_table_np[row_idx, :total_groups, :]
        old_flat = old_2d.reshape(-1).astype(np.int64)
        new_flat = new_counts.reshape(-1).astype(np.int64)
        col_axis = np.arange(row_view.shape[1])
        freed_mask = (col_axis[None, :] >= new_flat[:, None]) & (
            col_axis[None, :] < old_flat[:, None]
        )
        freed_ids = row_view[freed_mask].astype(np.int32, copy=True)
        row_view[freed_mask] = 0
        self.num_blocks_per_row[row_idx, :total_groups] = (
            new_counts.reshape(-1))
        return freed_ids

    def null_front_blocks_sliding(
        self,
        row_idx: int,
        sliding_layer_ids: np.ndarray,
        num_head_groups_per_layer: int,
        num_skipped_blocks: int,
    ) -> np.ndarray:
        """Free out-of-window front blocks of sliding-window layers in place.

        For a sliding-window layer, FlashAttention only attends to the last
        ``sliding_window`` tokens, so blocks holding tokens entirely before the
        window are never read and their physical blocks can be returned to the
        pool. Unlike ``compact_after_compress_all_layers`` (which keeps the
        front and shrinks the tail), this keeps the in-window TAIL and nulls the
        FRONT: the freed leading entries are set to the null block (id 0) while
        ``num_blocks_per_row`` stays unchanged, so the position -> block mapping
        of the surviving tail is preserved (no token-position remap, output
        unchanged).

        Args:
            row_idx: persistent-batch row of the request.
            sliding_layer_ids: physical indices of the sliding-window layers.
            num_head_groups_per_layer: groups (page columns) per layer; the
                group axis of the ragged block table is
                ``layer * num_head_groups_per_layer + group``.
            num_skipped_blocks: number of leading blocks now outside the window,
                ``(num_computed_tokens - sliding_window + 1) // block_size``;
                uniform across every sliding layer and group (single window).

        Returns:
            int32 ndarray of the freed physical block ids (null id 0 excluded),
            to be returned to the pool and null-in-placed in the manager's
            per-request array. Empty when there is nothing to free.
        """
        if num_skipped_blocks <= 0 or sliding_layer_ids.size == 0:
            return np.empty(0, dtype=np.int32)
        ngpl = num_head_groups_per_layer
        # Flat group rows of every sliding layer: layer*ngpl + [0..ngpl).
        group_rows = (
            sliding_layer_ids.astype(np.int64)[:, None] * ngpl
            + np.arange(ngpl, dtype=np.int64)[None, :]
        ).reshape(-1)
        # Fancy indexing returns a copy (rows are non-contiguous), so modify the
        # copy and assign it back. Clamp the freed range per row to its actual
        # fill so we never touch unallocated columns.
        row_view = self.block_table.np[row_idx, group_rows, :]
        counts = self.num_blocks_per_row[row_idx, group_rows]
        col_axis = np.arange(row_view.shape[1])
        skip = np.minimum(counts.astype(np.int64), num_skipped_blocks)
        front_mask = col_axis[None, :] < skip[:, None]
        freed = row_view[front_mask]
        freed = freed[freed != 0].astype(np.int32, copy=True)
        row_view[front_mask] = 0
        self.block_table.np[row_idx, group_rows, :] = row_view
        # ``num_blocks_per_row`` is intentionally left unchanged (the surviving
        # tail keeps its column positions).
        return freed
