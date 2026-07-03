# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)


class BlockTable:
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
        num_head_groups: int | None = None,
        num_head_groups_per_layer: int | None = None,
    ):
        """
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
            num_head_groups: Total head-groups
                (``num_head_groups_per_layer × num_layers``). When set,
                switches to ragged 3D mode; otherwise this is the
                base 2D layout.
            num_head_groups_per_layer: Per-layer head-group count
                (metadata for layer-aware consumers).
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        # Ragged paging lives on a separate code path so the base
        # 2D path stays a zero-overhead copy of the upstream layout.
        self.num_head_groups = num_head_groups
        self.ragged = num_head_groups is not None
        self.num_head_groups_per_layer = num_head_groups_per_layer
        if self.ragged:
            assert num_head_groups is not None and num_head_groups > 0, (
                "num_head_groups must be a positive int."
            )
            assert kernel_block_size == block_size, (
                "Ragged paging requires kernel_block_size == "
                "block_size (hybrid kernel block sizes not supported)."
            )
            # Default to ``num_head_groups`` so single-layer unit tests
            # can omit it; production callers always pass the real value.
            if num_head_groups_per_layer is None:
                num_head_groups_per_layer = num_head_groups
                self.num_head_groups_per_layer = num_head_groups_per_layer
            assert num_head_groups % num_head_groups_per_layer == 0, (
                "num_head_groups_per_layer must divide num_head_groups; "
                f"got num_head_groups={num_head_groups}, "
                f"num_head_groups_per_layer={num_head_groups_per_layer}.")

        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        if self.ragged:
            assert num_head_groups is not None  # for type-checkers
            self.block_table = self._make_buffer(
                self.max_num_reqs,
                num_head_groups,
                self.max_num_blocks_per_req,
                dtype=torch.int32,
            )
            self.num_blocks_per_row = np.zeros(
                (max_num_reqs, num_head_groups), dtype=np.int32
            )
            self.slot_mapping = self._make_buffer(
                num_head_groups,
                self.max_num_batched_tokens,
                dtype=torch.int64,
            )
        else:
            self.block_table = self._make_buffer(
                self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
            )
            self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

            self.slot_mapping = self._make_buffer(
                self.max_num_batched_tokens, dtype=torch.int64
            )

        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

        if self.ragged:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "Ragged paging is not implemented for "
                "DCP/PCP world sizes > 1."
            )

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return

        if self.ragged:
            self._append_row_grouped(block_ids, row_idx)
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

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

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        if self.ragged:
            self.num_blocks_per_row[row_idx, :] = 0
        else:
            self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

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
        assert self.ragged, "snapshot_row is ragged only."
        return (
            self.block_table.np[row_idx, :, :].copy(),
            self.num_blocks_per_row[row_idx, :].copy(),
        )

    def restore_row(
        self, row_idx: int, snapshot: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Write back a row captured by :meth:`snapshot_row`."""
        assert self.ragged, "restore_row is ragged only."
        block_ids_np, counts = snapshot
        self.block_table.np[row_idx, :, :] = block_ids_np
        self.num_blocks_per_row[row_idx, :] = counts

    def move_row(self, src: int, tgt: int) -> None:
        if self.ragged:
            block_table_np = self.block_table.np
            block_table_np[tgt, :, :] = block_table_np[src, :, :]
            self.num_blocks_per_row[tgt, :] = self.num_blocks_per_row[src, :]
            return
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        if self.ragged:
            block_table_np = self.block_table.np
            tmp = block_table_np[src, :, :].copy()
            block_table_np[src, :, :] = block_table_np[tgt, :, :]
            block_table_np[tgt, :, :] = tmp
            tmp_n = self.num_blocks_per_row[src, :].copy()
            self.num_blocks_per_row[src, :] = self.num_blocks_per_row[tgt, :]
            self.num_blocks_per_row[tgt, :] = tmp_n
            return
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        if self.ragged:
            self._compute_slot_mapping_grouped(req_indices, positions)
            return

        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        if total_cp_world_size > 1:
            # Note(hc): The DCP implement store kvcache with an interleave
            # style, the kvcache for the token whose token_idx is i is
            # always stored on the GPU whose dcp_rank equals i % cp_world_size:

            # Use a "virtual block" which equals to world_size * block_size
            # for block_table_indices calculation.
            virtual_block_size = self.block_size * total_cp_world_size
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + positions // virtual_block_size
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens.
            virtual_block_offsets = positions % virtual_block_size
            mask = (
                virtual_block_offsets
                // self.cp_kv_cache_interleave_size
                % total_cp_world_size
                == total_cp_rank
            )
            # Calculate local block_offsets
            block_offsets = (
                virtual_block_offsets
                // (total_cp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # Calculate slot_mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1
            )
        else:
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req + positions // self.block_size
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping.np[: req_indices.shape[0]],
            )

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

    def commit_block_table(self, num_reqs: int) -> None:
        if self.ragged:
            self.block_table.gpu[:num_reqs].copy_(
                self.block_table.cpu[:num_reqs], non_blocking=True
            )
            return
        self.block_table.copy_to_gpu(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        if self.ragged:
            self.slot_mapping.gpu[:, :num_tokens].copy_(
                self.slot_mapping.cpu[:, :num_tokens], non_blocking=True
            )
            return
        self.slot_mapping.copy_to_gpu(num_tokens)

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)
        # Reset row counts so a subsequent append_row writes from offset 0.
        self.num_blocks_per_row.fill(0)

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
        assert self.ragged, (
            "compact_after_compress_all_layers is ragged only.")
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
        assert self.ragged, (
            "null_front_blocks_sliding is ragged only.")
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

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
        num_head_groups: int | None = None,
        num_head_groups_per_layer: int | None = None,
    ) -> None:
        # Note(hc): each dcp rank only store
        # (max_model_len//dcp_world_size) tokens in kvcache,
        # so the block_size which used for calc max_num_blocks_per_req
        # must be multiplied by dcp_world_size.
        try:
            pcp_world_size = get_pcp_group().world_size
        except AssertionError:
            # PCP might not be initialized in testing
            pcp_world_size = 1
        try:
            dcp_world_size = get_dcp_group().world_size
        except AssertionError:
            # DCP might not be initialized in testing
            dcp_world_size = 1

        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        total_cp_world_size = dcp_world_size * pcp_world_size

        # Ragged mode collapses the underlying list to a single 3D
        # BlockTable. ``num_head_groups`` is the total (per_layer × layers);
        # the layer dimension is already absorbed into the flat group
        # index, so ``append_row`` does no per-layer tiling.
        self.ragged = num_head_groups is not None
        self.num_head_groups = num_head_groups
        self.num_head_groups_per_layer = num_head_groups_per_layer
        if self.ragged:
            assert len(block_sizes) == 1, (
                "Ragged paging requires a single KVCacheGroupSpec; "
                f"got {len(block_sizes)} block_sizes.")
            assert num_head_groups is not None
            if num_head_groups_per_layer is not None:
                assert num_head_groups % num_head_groups_per_layer == 0, (
                    f"num_head_groups ({num_head_groups}) must be divisible "
                    f"by num_head_groups_per_layer "
                    f"({num_head_groups_per_layer}).")

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max(
                    cdiv(max_model_len, block_size * total_cp_world_size),
                    1 + num_speculative_tokens,
                ),
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
                num_head_groups=num_head_groups,
                num_head_groups_per_layer=num_head_groups_per_layer,
            )
            for block_size, kernel_block_size in zip(block_sizes, kernel_block_sizes)
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        if self.ragged:
            # Caller delivers ``num_head_groups × num_new`` ids in
            # group-major order; a single ``BlockPool.get_new_blocks``
            # call already covered every (layer, head-group) pair.
            self.block_tables[0].append_row(block_ids[0], row_idx)
            return
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        if self.ragged:
            self.block_tables[0].add_row(block_ids[0], row_idx)
            return
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def snapshot_row(self, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Ragged paging keeps every layer/group in ``block_tables[0]``."""
        assert self.ragged, "snapshot_row is ragged only."
        return self.block_tables[0].snapshot_row(row_idx)

    def restore_row(
        self, row_idx: int, snapshot: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Ragged paging keeps every layer/group in ``block_tables[0]``."""
        assert self.ragged, "restore_row is ragged only."
        self.block_tables[0].restore_row(row_idx, snapshot)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_slot_mapping(num_tokens)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
