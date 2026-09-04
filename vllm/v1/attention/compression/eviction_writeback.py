# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Moving the kept KV of every evicted (layer, head-group) at one boundary.

The executor decides WHAT survives; this module moves it. Two backends share
one :class:`EvictionPlan`:

* :class:`TorchWriteback` -- the reference. One cluster at a time, plain
  torch indexing, runs anywhere. This is the behaviour every other backend
  must reproduce bit for bit.
* :class:`TritonWriteback` -- four launches for the whole boundary, however
  many clusters it has. A sort kernel orders each column's top-k positions in
  place, then three write-back kernels compact the kept KV inside its own
  pages, with no temporary copy of the KV.

Why the in-place kernels are safe. Within one column the kept positions are
strictly increasing, so the j-th kept position is >= j: a destination slot
never lies past its source. Every destination is below ``kept``, so a source
at or past ``kept`` can never be overwritten -- those are moved by a fully
parallel kernel. The sources below ``kept`` are the ENDANGERED ones; because
the positions ascend they feed a prefix of the destination, and one program
per (cluster, column, K|V) walks that prefix in ascending tiles, reading a
whole tile before writing it: later tiles' sources are >= the tile's end. The
prefix goes first, the parallel remainder second, and the zero padding of the
last block last, since a padded slot may itself be a source. Programs never
share pages, so no ordering between them is needed.

Slots below ``sink_size + locked`` are kept at the same position, so the
kernels start past them; the reference rewrites them with their own values.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch

from vllm.triton_utils import tl, triton


class PlanCol(IntEnum):
    """Columns of :attr:`EvictionPlan.table`, one row per evicted cluster."""
    #: Physical layer: index into the per-layer KV cache list.
    LAYER = 0
    #: Element offset of the cluster's row in the ragged block table.
    BT_OFFSET = 1
    #: ``compressed_layer * num_groups + group``: row into ``sorted_idx``.
    CLUSTER = 2
    #: ``sink_size + locked``: first slot whose content changes.
    KEPT_LO = 3
    #: Count of top-k positions taken from ``sorted_idx``.
    K_ALIGNED = 4
    #: First position of the always-kept tail.
    TAIL_LO = 5
    #: Positions that survive; the write-back zero-pads to a block multiple.
    KEPT = 6


NUM_PLAN_COLS = len(PlanCol)


@dataclass
class EvictionPlan:
    """Per-cluster geometry of one boundary's eviction, in CPU memory."""
    #: ``[n, NUM_PLAN_COLS]`` int64, see :class:`PlanCol`.
    table: np.ndarray
    sink_size: int
    tail_size: int
    #: Rectangular eval width: every top-k position is below it.
    eval_len: int

    @property
    def num_clusters(self) -> int:
        return int(self.table.shape[0])


class PositionReport(IntEnum):
    """Whether ``run`` returns each cluster's kept positions."""
    SKIP = 0
    RETURN = 1


def gather_and_writeback_kept_kv(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    block_size: int,
    sink_idx: torch.Tensor,
    locked: int,
    k_aligned: int,
    kept_lo: int,
    sorted_idx_group: torch.Tensor | None,
    tail_idx: torch.Tensor,
    tail_lo: int,
    kept_length: int,
) -> torch.Tensor:
    """Evict one (layer, group) with torch indexing: the reference.

    Gathers the kept positions straight from the pages -- O(kept), the
    cluster's KV is never built as one slab -- and writes them back
    block-aligned into the same blocks, trailing partial block zero-padded.

    The kept positions form a ``[page_group_size, kept_length]`` matrix in
    which every column keeps the SAME sink / locked / tail and only the middle
    ``k_aligned`` span differs. It is returned so the caller can apply the
    identical positions to anything else stored per cache slot.
    ``sorted_idx_group`` is dereferenced only when ``k_aligned > 0``.
    """
    device = kv_cache.device
    page_group_size = kv_cache.shape[2]
    head_size = kv_cache.shape[4]
    sink_size = int(sink_idx.numel())

    col_parts: list[torch.Tensor] = []
    if sink_size > 0:
        col_parts.append(sink_idx.unsqueeze(0).expand(page_group_size, -1))
    if locked > 0:
        col_parts.append(
            torch.arange(sink_size, sink_size + locked, device=device,
                         dtype=torch.long)
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
        else torch.empty(page_group_size, 0, dtype=torch.long, device=device))

    keep_block = torch.div(keep_mat, block_size, rounding_mode="floor")
    keep_offset = keep_mat - keep_block * block_size
    col_ix = torch.arange(
        page_group_size, device=device, dtype=torch.long
    ).unsqueeze(1).expand_as(keep_mat)
    # ``block_ids[keep_block]`` picks the page, ``col_ix`` the column,
    # ``keep_offset`` the token -> [2, page_group_size, kept, head_size].
    kept_kv = kv_cache[
        :, block_ids[keep_block], col_ix, keep_offset
    ].permute(0, 2, 1, 3).contiguous()

    n_blocks_write = (kept_length + block_size - 1) // block_size
    padded_size = n_blocks_write * block_size
    if kept_length < padded_size:
        pad = torch.zeros(
            2, padded_size - kept_length, page_group_size, head_size,
            dtype=kept_kv.dtype, device=device)
        kept_kv = torch.cat([kept_kv, pad], dim=1)
    # Inverse of the gather: token-major slab back to column-major pages.
    kv_cache[:, block_ids[:n_blocks_write]] = (
        kept_kv.view(2, n_blocks_write, block_size, page_group_size, head_size)
        .permute(0, 1, 3, 2, 4)
    )
    return keep_mat


class KeptKVWriteback:
    """Interface both backends implement."""

    def run(
        self,
        plan: EvictionPlan,
        layer_kv_caches: list[torch.Tensor],
        block_table_gpu: torch.Tensor,
        sorted_idx: torch.Tensor,
        report: PositionReport,
    ) -> list[torch.Tensor] | None:
        """Apply ``plan`` to the caches.

        ``sorted_idx`` is ``[num_clusters_total, page_group_size, width]``
        with each cluster's top-k positions first, in score order; a backend
        may reorder those first ``K_ALIGNED`` entries. With
        ``PositionReport.RETURN`` the result holds, per plan row, the
        ``[page_group_size, KEPT]`` int64 source positions the write-back
        used, in slot order.
        """
        raise NotImplementedError


class TorchWriteback(KeptKVWriteback):
    """Reference: one cluster at a time through
    :func:`gather_and_writeback_kept_kv`."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size

    def run(self, plan, layer_kv_caches, block_table_gpu, sorted_idx, report):
        block_size = self.block_size
        device = layer_kv_caches[0].device
        sink_idx = torch.arange(plan.sink_size, device=device, dtype=torch.long)
        tail_idx = torch.arange(plan.tail_size, device=device, dtype=torch.long)
        bt_flat = block_table_gpu.reshape(-1)
        positions: list[torch.Tensor] = []

        for row in plan.table:
            kept = int(row[PlanCol.KEPT])
            kept_lo = int(row[PlanCol.KEPT_LO])
            k_aligned = int(row[PlanCol.K_ALIGNED])
            tail_lo = int(row[PlanCol.TAIL_LO])
            total_seen = tail_lo + plan.tail_size
            n_blocks = (total_seen + block_size - 1) // block_size
            bt_offset = int(row[PlanCol.BT_OFFSET])
            block_ids = bt_flat[bt_offset:bt_offset + n_blocks].long()

            keep_mat = gather_and_writeback_kept_kv(
                kv_cache=layer_kv_caches[int(row[PlanCol.LAYER])],
                block_ids=block_ids,
                block_size=block_size,
                sink_idx=sink_idx,
                locked=kept_lo - plan.sink_size,
                k_aligned=k_aligned,
                kept_lo=kept_lo,
                sorted_idx_group=(
                    sorted_idx[int(row[PlanCol.CLUSTER])]
                    if k_aligned > 0 else None),
                tail_idx=tail_idx,
                tail_lo=tail_lo,
                kept_length=kept,
            )
            if report == PositionReport.RETURN:
                positions.append(keep_mat)

        return positions if report == PositionReport.RETURN else None


# --- Triton -----------------------------------------------------------------

#: Positions per step of the in-place sort's scan.
_SORT_BLOCK = 1024
#: Destination slots one program of the sequential kernel moves per step.
_PREFIX_TOKEN_TILE = 4
#: Destination slots one program of the parallel kernel moves.
_TOKEN_TILE = 32
#: Widest head slice one program moves at a time.
_HEAD_TILE = 128
#: Warps per program of the write-back kernels.
_NUM_WARPS = 1


# ``eval_len`` and ``num_tiles`` change from boundary to boundary; specialising
# on their value would recompile the kernel for every new one.
@triton.jit(do_not_specialize=["eval_len"])
def _sort_kept_positions_kernel(
    plan_ptr,
    sorted_ptr,
    mask_ptr,
    endangered_ptr,
    width,
    eval_len,
    PAGE_GROUP_SIZE: tl.constexpr,
    NUM_COLS: tl.constexpr,
    K_ALIGNED_COL: tl.constexpr,
    CLUSTER_COL: tl.constexpr,
    KEPT_LO_COL: tl.constexpr,
    KEPT_COL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sort one column's first ``k`` top-k positions ascending, in place.

    Radix-free: mark each selected position in a byte mask over
    ``[0, eval_len)``, then scan the mask once and write the marked positions
    back over the unsorted run. Three passes of a single program; a barrier
    separates each so the writes of one pass are visible to the next.

    The scan also counts the selected positions below ``kept`` -- the ones a
    destination write could clobber -- and adds them to ``endangered[row,
    col]``, which the caller pre-filled with the tail's share.
    """
    row = tl.program_id(0)
    col = tl.program_id(1)
    k = tl.load(plan_ptr + row * NUM_COLS + K_ALIGNED_COL)
    if k == 0:
        return
    cluster = tl.load(plan_ptr + row * NUM_COLS + CLUSTER_COL)
    kept_span = (tl.load(plan_ptr + row * NUM_COLS + KEPT_COL)
                 - tl.load(plan_ptr + row * NUM_COLS + KEPT_LO_COL))
    base = (cluster * PAGE_GROUP_SIZE + col) * width
    idx_base = sorted_ptr + base
    mask_base = mask_ptr + base
    offs = tl.arange(0, BLOCK)

    for start in range(0, eval_len, BLOCK):
        pos = start + offs
        tl.store(mask_base + pos, tl.zeros([BLOCK], dtype=tl.uint8),
                 mask=pos < eval_len)
    tl.debug_barrier()

    for start in range(0, k, BLOCK):
        i = start + offs
        valid = i < k
        pos = tl.load(idx_base + i, mask=valid, other=0)
        tl.store(mask_base + pos, tl.full([BLOCK], 1, tl.uint8), mask=valid)
    tl.debug_barrier()

    placed = tl.zeros([], dtype=tl.int32)
    below_kept = tl.zeros([], dtype=tl.int32)
    for start in range(0, eval_len, BLOCK):
        pos = start + offs
        in_range = pos < eval_len
        marked = tl.load(mask_base + pos, mask=in_range, other=0).to(tl.int32)
        rank = placed + tl.cumsum(marked, axis=0) - 1
        tl.store(idx_base + rank, pos.to(tl.int64), mask=(marked != 0))
        placed += tl.sum(marked, axis=0)
        below_kept += tl.sum(tl.where(pos < kept_span, marked, 0), axis=0)
    endangered = endangered_ptr + row * PAGE_GROUP_SIZE + col
    tl.store(endangered, tl.load(endangered) + below_kept)


@triton.jit
def _cluster_column(
    plan_ptr, kv_ptrs, kv_dtype_ptr, sorted_ptr, bt_ptr, endangered_ptr,
    width, stride_col, num_tiles,
    PAGE_GROUP_SIZE: tl.constexpr, NUM_COLS: tl.constexpr,
    LAYER_COL: tl.constexpr, BT_OFFSET_COL: tl.constexpr,
    CLUSTER_COL: tl.constexpr, KEPT_LO_COL: tl.constexpr,
    K_ALIGNED_COL: tl.constexpr, TAIL_LO_COL: tl.constexpr,
    KEPT_COL: tl.constexpr, HEAD_SPLITS: tl.constexpr,
):
    """Resolve this program's (cluster, column, K|V, head slice, tile) from
    the grid and the plan. Shared by the three write-back kernels; a grid has
    three axes, so K|V, head slice and tile share the last one."""
    row = tl.program_id(0)
    col = tl.program_id(1)
    kv_and_split = tl.program_id(2) // num_tiles
    tile = tl.program_id(2) % num_tiles
    which_kv = kv_and_split // HEAD_SPLITS
    head_split = kv_and_split % HEAD_SPLITS

    plan = plan_ptr + row * NUM_COLS
    layer = tl.load(plan + LAYER_COL)
    bt = bt_ptr + tl.load(plan + BT_OFFSET_COL)
    cluster = tl.load(plan + CLUSTER_COL)
    kept_lo = tl.load(plan + KEPT_LO_COL)
    k_aligned = tl.load(plan + K_ALIGNED_COL)
    tail_lo = tl.load(plan + TAIL_LO_COL)
    kept = tl.load(plan + KEPT_COL)
    endangered = tl.load(endangered_ptr + row * PAGE_GROUP_SIZE + col)

    # One pointer per (layer, K|V) plane: the plane stride of a large pool
    # exceeds 32 bits, and an integer argument that changes width between the
    # warm-up and the real cache would compile a second variant.
    kv = tl.load(kv_ptrs + layer * 2 + which_kv).to(
        tl.pointer_type(kv_dtype_ptr.dtype.element_ty))
    kv = kv + col * stride_col
    mid = sorted_ptr + (cluster * PAGE_GROUP_SIZE + col) * width
    return (kv, bt, mid, kept_lo, k_aligned, tail_lo, kept, endangered,
            head_split, tile)


@triton.jit
def _load_tile(kv, bt, mid, kept_lo, mid_end, k_aligned, tail_lo, kept,
               j, h, h_ok, stride_block, stride_tok, BLOCK_SIZE: tl.constexpr):
    """The KV that destination slots ``j`` receive; zeros past ``kept``."""
    live = j < kept
    in_mid = j < mid_end
    mid_i = tl.minimum(j - kept_lo, k_aligned - 1)
    mid_pos = tl.load(mid + mid_i, mask=live & in_mid, other=0) + kept_lo
    src = tl.where(in_mid, mid_pos, tail_lo + (j - mid_end))
    src_page = tl.load(bt + src // BLOCK_SIZE, mask=live, other=0)
    src_off = (src_page.to(tl.int64) * stride_block
               + (src % BLOCK_SIZE) * stride_tok)
    return tl.load(kv + src_off[:, None] + h[None, :],
                   mask=live[:, None] & h_ok[None, :], other=0)


@triton.jit
def _store_tile(kv, bt, vals, j, j_ok, h, h_ok, stride_block, stride_tok,
                BLOCK_SIZE: tl.constexpr):
    page = tl.load(bt + j // BLOCK_SIZE, mask=j_ok, other=0)
    off = page.to(tl.int64) * stride_block + (j % BLOCK_SIZE) * stride_tok
    tl.store(kv + off[:, None] + h[None, :], vals,
             mask=j_ok[:, None] & h_ok[None, :])


@triton.jit(do_not_specialize=["num_tiles"])
def _writeback_prefix_kernel(
    plan_ptr, kv_ptrs, kv_dtype_ptr, sorted_ptr, bt_ptr, endangered_ptr,
    width, head_size, stride_block, stride_col, stride_tok, num_tiles,
    PAGE_GROUP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    NUM_COLS: tl.constexpr, LAYER_COL: tl.constexpr,
    BT_OFFSET_COL: tl.constexpr, CLUSTER_COL: tl.constexpr,
    KEPT_LO_COL: tl.constexpr, K_ALIGNED_COL: tl.constexpr,
    TAIL_LO_COL: tl.constexpr, KEPT_COL: tl.constexpr,
    HEAD_SPLITS: tl.constexpr, TOKEN_TILE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
):
    """Move the endangered prefix ``[kept_lo, kept_lo + endangered)`` of one
    (cluster, column, K|V, head slice), one ascending tile at a time. Runs
    first and alone; see the module docstring."""
    (kv, bt, mid, kept_lo, k_aligned, tail_lo, kept, endangered,
     head_split, tile) = _cluster_column(
        plan_ptr, kv_ptrs, kv_dtype_ptr, sorted_ptr, bt_ptr, endangered_ptr,
        width, stride_col, num_tiles,
        PAGE_GROUP_SIZE, NUM_COLS, LAYER_COL,
        BT_OFFSET_COL, CLUSTER_COL, KEPT_LO_COL, K_ALIGNED_COL, TAIL_LO_COL,
        KEPT_COL, HEAD_SPLITS)
    mid_end = kept_lo + k_aligned
    h = head_split * HEAD_TILE + tl.arange(0, HEAD_TILE)
    h_ok = h < head_size
    t = tl.arange(0, TOKEN_TILE)

    prefix_end = kept_lo + endangered
    for j0 in range(kept_lo, prefix_end, TOKEN_TILE):
        j = j0 + t
        # The last tile may reach past the prefix; those slots belong to the
        # parallel kernel, so they are loaded (harmless) but not stored.
        j_ok = j < prefix_end
        vals = _load_tile(kv, bt, mid, kept_lo, mid_end, k_aligned, tail_lo,
                          kept, j, h, h_ok, stride_block, stride_tok,
                          BLOCK_SIZE)
        _store_tile(kv, bt, vals, j, j_ok, h, h_ok, stride_block, stride_tok,
                    BLOCK_SIZE)


@triton.jit(do_not_specialize=["num_tiles"])
def _writeback_parallel_kernel(
    plan_ptr, kv_ptrs, kv_dtype_ptr, sorted_ptr, bt_ptr, endangered_ptr,
    width, head_size, stride_block, stride_col, stride_tok, num_tiles,
    PAGE_GROUP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    NUM_COLS: tl.constexpr, LAYER_COL: tl.constexpr,
    BT_OFFSET_COL: tl.constexpr, CLUSTER_COL: tl.constexpr,
    KEPT_LO_COL: tl.constexpr, K_ALIGNED_COL: tl.constexpr,
    TAIL_LO_COL: tl.constexpr, KEPT_COL: tl.constexpr,
    HEAD_SPLITS: tl.constexpr, TOKEN_TILE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
):
    """Move one tile of the remainder ``[kept_lo + endangered, kept)``: its
    sources are all at or past ``kept``, which no destination touches."""
    (kv, bt, mid, kept_lo, k_aligned, tail_lo, kept, endangered,
     head_split, tile) = _cluster_column(
        plan_ptr, kv_ptrs, kv_dtype_ptr, sorted_ptr, bt_ptr, endangered_ptr,
        width, stride_col, num_tiles,
        PAGE_GROUP_SIZE, NUM_COLS, LAYER_COL,
        BT_OFFSET_COL, CLUSTER_COL, KEPT_LO_COL, K_ALIGNED_COL, TAIL_LO_COL,
        KEPT_COL, HEAD_SPLITS)
    j0 = kept_lo + endangered + tile * TOKEN_TILE
    if j0 >= kept:
        return
    mid_end = kept_lo + k_aligned
    h = head_split * HEAD_TILE + tl.arange(0, HEAD_TILE)
    h_ok = h < head_size
    j = j0 + tl.arange(0, TOKEN_TILE)
    j_ok = j < kept
    vals = _load_tile(kv, bt, mid, kept_lo, mid_end, k_aligned, tail_lo, kept,
                      j, h, h_ok, stride_block, stride_tok, BLOCK_SIZE)
    _store_tile(kv, bt, vals, j, j_ok, h, h_ok, stride_block, stride_tok,
                BLOCK_SIZE)


@triton.jit(do_not_specialize=["num_tiles"])
def _writeback_pad_kernel(
    plan_ptr, kv_ptrs, kv_dtype_ptr, sorted_ptr, bt_ptr, endangered_ptr,
    width, head_size, stride_block, stride_col, stride_tok, num_tiles,
    PAGE_GROUP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    NUM_COLS: tl.constexpr, LAYER_COL: tl.constexpr,
    BT_OFFSET_COL: tl.constexpr, CLUSTER_COL: tl.constexpr,
    KEPT_LO_COL: tl.constexpr, K_ALIGNED_COL: tl.constexpr,
    TAIL_LO_COL: tl.constexpr, KEPT_COL: tl.constexpr,
    HEAD_SPLITS: tl.constexpr, HEAD_TILE: tl.constexpr,
):
    """Zero ``[kept, next block boundary)`` once every source has been read."""
    (kv, bt, mid, kept_lo, k_aligned, tail_lo, kept, endangered,
     head_split, tile) = _cluster_column(
        plan_ptr, kv_ptrs, kv_dtype_ptr, sorted_ptr, bt_ptr, endangered_ptr,
        width, stride_col, num_tiles,
        PAGE_GROUP_SIZE, NUM_COLS, LAYER_COL,
        BT_OFFSET_COL, CLUSTER_COL, KEPT_LO_COL, K_ALIGNED_COL, TAIL_LO_COL,
        KEPT_COL, HEAD_SPLITS)
    end = ((kept + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    h = head_split * HEAD_TILE + tl.arange(0, HEAD_TILE)
    h_ok = h < head_size
    j = kept + tl.arange(0, BLOCK_SIZE)
    j_ok = j < end
    zeros = tl.zeros([BLOCK_SIZE, HEAD_TILE], dtype=kv.dtype.element_ty)
    _store_tile(kv, bt, zeros, j, j_ok, h, h_ok, stride_block, stride_tok,
                BLOCK_SIZE)


#: int64 elements per 16-byte alignment unit.
_ALIGN_INT64 = 2


def _round_up(n: int, unit: int) -> int:
    return (n + unit - 1) // unit * unit


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


class TritonWriteback(KeptKVWriteback):
    """Whole-boundary write-back in four launches: sort, endangered prefix,
    parallel remainder, zero padding.

    ``keep_mask`` is scratch for the sort: ``[num_clusters_total,
    page_group_size, width]`` uint8 matching ``sorted_idx``, allocated once by
    the caller. A boundary allocates only its few-KB plan table.
    """

    def __init__(
        self,
        block_size: int,
        page_group_size: int,
        head_size: int,
        keep_mask: torch.Tensor,
    ) -> None:
        self.block_size = block_size
        self.page_group_size = page_group_size
        self.head_size = head_size
        self.keep_mask = keep_mask
        self.prefix_token_tile = _PREFIX_TOKEN_TILE
        self.token_tile = _TOKEN_TILE
        self.num_warps = _NUM_WARPS
        self.head_tile = min(_next_pow2(head_size), _HEAD_TILE)
        self.head_splits = (head_size + self.head_tile - 1) // self.head_tile

    def run(self, plan, layer_kv_caches, block_table_gpu, sorted_idx, report):
        if plan.num_clusters == 0:
            return [] if report == PositionReport.RETURN else None
        device = layer_kv_caches[0].device
        kv0 = layer_kv_caches[0]
        assert sorted_idx.dim() == 3 and sorted_idx.is_contiguous()
        assert self.keep_mask.shape == sorted_idx.shape
        width = sorted_idx.shape[-1]
        n = plan.num_clusters
        pg = self.page_group_size
        cols = plan.table

        # Endangered sources start as the tail's share -- the tail positions
        # below ``kept`` -- uniform over columns; the sort adds each column's
        # selected positions below ``kept``. One transfer carries both tables.
        tail_below_kept = np.clip(
            cols[:, PlanCol.KEPT] - cols[:, PlanCol.TAIL_LO], 0,
            plan.tail_size)
        # Segments are padded to 16 bytes: Triton compiles a separate variant
        # per pointer alignment, and one warm-up must cover every boundary.
        plane_bytes = kv0.stride(0) * kv0.element_size()
        segments = [
            cols.reshape(-1),
            np.repeat(tail_below_kept, pg),
            np.array([kv.data_ptr() + which * plane_bytes
                      for kv in layer_kv_caches for which in (0, 1)],
                     dtype=np.int64),
        ]
        starts = []
        offset = 0
        for seg in segments:
            starts.append(offset)
            offset += _round_up(seg.size, _ALIGN_INT64)
        host = np.zeros(offset, dtype=np.int64)
        for seg, start in zip(segments, starts):
            host[start:start + seg.size] = seg
        dev = torch.from_numpy(host).to(device, non_blocking=True)
        table = dev[starts[0]:starts[0] + n * NUM_PLAN_COLS].view(
            n, NUM_PLAN_COLS)
        endangered = dev[starts[1]:starts[1] + n * pg]
        kv_ptrs = dev[starts[2]:starts[2] + 2 * len(layer_kv_caches)]

        if plan.eval_len > 0:
            _sort_kept_positions_kernel[(n, pg)](
                table, sorted_idx, self.keep_mask, endangered, width,
                plan.eval_len,
                PAGE_GROUP_SIZE=pg, NUM_COLS=NUM_PLAN_COLS,
                K_ALIGNED_COL=int(PlanCol.K_ALIGNED),
                CLUSTER_COL=int(PlanCol.CLUSTER),
                KEPT_LO_COL=int(PlanCol.KEPT_LO), KEPT_COL=int(PlanCol.KEPT),
                BLOCK=_SORT_BLOCK,
            )

        common = dict(
            PAGE_GROUP_SIZE=pg, BLOCK_SIZE=self.block_size,
            NUM_COLS=NUM_PLAN_COLS,
            LAYER_COL=int(PlanCol.LAYER), BT_OFFSET_COL=int(PlanCol.BT_OFFSET),
            CLUSTER_COL=int(PlanCol.CLUSTER), KEPT_LO_COL=int(PlanCol.KEPT_LO),
            K_ALIGNED_COL=int(PlanCol.K_ALIGNED),
            TAIL_LO_COL=int(PlanCol.TAIL_LO), KEPT_COL=int(PlanCol.KEPT),
            HEAD_SPLITS=self.head_splits, HEAD_TILE=self.head_tile,
            num_warps=self.num_warps,
        )
        args = (table, kv_ptrs, kv0, sorted_idx, block_table_gpu, endangered,
                width, self.head_size,
                kv0.stride(1), kv0.stride(2), kv0.stride(3))
        planes = 2 * self.head_splits
        max_span = int((cols[:, PlanCol.KEPT] - cols[:, PlanCol.KEPT_LO]).max())
        tiles = (max_span + self.token_tile - 1) // self.token_tile

        _writeback_prefix_kernel[(n, pg, planes)](
            *args, 1, TOKEN_TILE=self.prefix_token_tile, **common)
        if tiles > 0:
            _writeback_parallel_kernel[(n, pg, planes * tiles)](
                *args, tiles, TOKEN_TILE=self.token_tile, **common)
        _writeback_pad_kernel[(n, pg, planes)](*args, 1, **common)

        if report == PositionReport.SKIP:
            return None
        return self._positions(plan, table, sorted_idx)

    def _positions(
        self,
        plan: EvictionPlan,
        table: torch.Tensor,
        sorted_idx: torch.Tensor,
    ) -> list[torch.Tensor]:
        """The ``[page_group_size, KEPT]`` source positions per plan row,
        rebuilt from the sorted middle span with the kernel's arithmetic."""
        pg = self.page_group_size
        width = sorted_idx.shape[-1]
        kept_max = int(plan.table[:, PlanCol.KEPT].max())
        j = torch.arange(kept_max, device=table.device,
                         dtype=torch.long).view(1, 1, -1)
        kept_lo = table[:, PlanCol.KEPT_LO].view(-1, 1, 1)
        k_aligned = table[:, PlanCol.K_ALIGNED].view(-1, 1, 1)
        tail_lo = table[:, PlanCol.TAIL_LO].view(-1, 1, 1)
        mid_end = kept_lo + k_aligned

        # One flat gather; indexing the cluster rows first would copy them.
        mid_i = torch.minimum((j - kept_lo).clamp(min=0),
                              (k_aligned - 1).clamp(min=0))
        col = torch.arange(pg, device=table.device,
                           dtype=torch.long).view(1, -1, 1)
        row_base = (table[:, PlanCol.CLUSTER].view(-1, 1, 1) * pg + col) * width
        mid_pos = sorted_idx.view(-1)[row_base + mid_i] + kept_lo
        pos = torch.where(j < kept_lo, j,
                          torch.where(j < mid_end, mid_pos,
                                      tail_lo + (j - mid_end)))
        return [pos[i, :, :int(row[PlanCol.KEPT])]
                for i, row in enumerate(plan.table)]

    def warmup(self, layer_kv_cache: torch.Tensor) -> None:
        """Compile both kernels now, on a two-block cluster kept whole, so the
        first real boundary pays no JIT. ``layer_kv_cache`` supplies device,
        dtype and strides; its first two pages are read and rewritten
        unchanged."""
        block_size = self.block_size
        device = layer_kv_cache.device
        total = 2 * block_size
        table = np.array([[0, 0, 0, 0, total, total, total]], dtype=np.int64)
        plan = EvictionPlan(table=table, sink_size=0, tail_size=0,
                            eval_len=total)
        block_table = torch.arange(2, device=device, dtype=torch.int32)
        sorted_idx = torch.empty_like(self.keep_mask, dtype=torch.int64)
        sorted_idx[0, :, :total] = torch.arange(total, device=device).flip(0)
        self.run(plan, [layer_kv_cache], block_table, sorted_idx,
                 PositionReport.RETURN)


def make_kept_kv_writeback(
    device: torch.device,
    block_size: int,
    page_group_size: int,
    head_size: int,
    keep_mask: torch.Tensor | None,
) -> KeptKVWriteback:
    """The Triton backend on CUDA, the torch reference elsewhere."""
    if device.type != "cuda" or keep_mask is None:
        return TorchWriteback(block_size)
    return TritonWriteback(block_size, page_group_size, head_size, keep_mask)
