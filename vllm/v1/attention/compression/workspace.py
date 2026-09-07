# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preallocated GPU memory for the KV-compression keep decision.

Every decision-sized tensor is allocated once from the configuration alone, so
nothing here allocates while requests are in flight. Growing on demand would
scale with how many requests happen to be mid-prefill, which no startup check
can bound.

The reservation lands in the weights term of the worker's memory profile, so
the KV pool shrinks by it and a configuration too large for the device fails at
startup rather than mid-generation.

Rows come from an allocator private to this class, not the ``InputBatch`` row
index: the input batch compacts its rows when a request finishes, which would
hand one request's buffers to another.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.attention.compression.budget_scope import (
    POOLED_BUDGET_SCOPES,
)
from vllm.v1.attention.compression.scorer import QK_SCORERS
from vllm.v1.attention.compression.slot_scores import (
    slot_scores_persist_across_steps,
)

if TYPE_CHECKING:
    from vllm.config.cache import CacheConfig

logger = init_logger(__name__)

_MIB = 1024.0 * 1024.0


@dataclass(frozen=True)
class WorkspaceSpec:
    """Shapes the workspace is built from, all known at startup.

    ``eval_capacity`` and ``slot_capacity`` are the WIDEST the active regime
    can produce, and its geometry is checked against them at runtime, so an
    overflowing shape raises instead of silently reallocating.
    """
    num_layers: int
    num_kv_heads: int          # per tensor-parallel rank
    num_groups: int            # head groups per layer
    page_group_size: int
    max_num_reqs: int
    chunk_size: int
    window_size: int
    #: Widest eval region (the positions a single keep decision may rank).
    eval_capacity: int
    #: Widest live cache length per (layer, group); 0 when the regime keeps no
    #: per-position statistics (the ratio regime).
    slot_capacity: int
    #: Most one entry may hold after a keep decision, 0 under the ratio regime.
    #: ``budget`` when the scope holds every entry to it, ``max_model_len`` when
    #: it pools -- one entry may then take the whole span, so only the physical
    #: limit bounds it. The keep decision reads this back as its ceiling, so
    #: buffer width and decision cannot disagree.
    per_group_capacity: int
    #: Budget scope ``per_group_capacity`` was derived from, so the compressor
    #: can refuse a scope the workspace was not sized for: a pooling scope on a
    #: ``uniform``-sized buffer silently caps every entry at ``budget`` again.
    budget_scope: str
    #: Rows of per-position statistics: ``max_num_reqs`` when the source keeps
    #: scores between steps, 1 when it rewrites every live slot, 0 when none.
    slot_rows: int
    #: dtype the active scorer produces, held exactly: the gate emits the model
    #: dtype and qk scorers float32, and rounding would change the decision.
    score_dtype: torch.dtype

    @staticmethod
    def from_cache_config(
        cache_config: "CacheConfig",
        num_layers: int,
        num_kv_heads: int,
        max_num_reqs: int,
        max_model_len: int,
        model_dtype: torch.dtype,
    ) -> "WorkspaceSpec":
        """``from_config`` with the knobs read off ``CacheConfig``."""
        return WorkspaceSpec.from_config(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            num_groups=num_kv_heads // cache_config.page_group_size,
            page_group_size=cache_config.page_group_size,
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            model_dtype=model_dtype,
            chunk_size=cache_config.compression_chunk_size,
            window_size=cache_config.compression_window_size,
            n_sink_tokens=cache_config.compression_n_sink_tokens,
            budget_tokens=cache_config.compression_budget_tokens,
            evict_current_chunk=cache_config.compression_evict_current_chunk,
            budget_scope=cache_config.compression_budget_scope,
            scorer=cache_config.compression_scorer,
            slot_score_source=cache_config.compression_slot_score_source,
        )

    @staticmethod
    def from_config(
        num_layers: int,
        num_kv_heads: int,
        num_groups: int,
        page_group_size: int,
        max_num_reqs: int,
        max_model_len: int,
        model_dtype: torch.dtype,
        chunk_size: int,
        window_size: int,
        n_sink_tokens: int,
        budget_tokens: int | None,
        evict_current_chunk: bool,
        scorer: str,
        slot_score_source: str,
        budget_scope: str = "uniform",
    ) -> "WorkspaceSpec":
        """Derive the shapes from the cache configuration.

        The bounds the regimes guarantee, each asserted at runtime rather than
        trusted. Under the ratio regime the eval region is the previous window
        plus the fresh chunk, never over one chunk, and no statistics are kept.
        Under a budget every entry is held to ``per_group_capacity``, so the
        live length before the next eviction is at most that plus a chunk, and
        the eval region is the live cache minus sink and protected tail.

        That capacity is ``budget`` under ``uniform``, which caps each entry,
        and ``max_model_len`` under a POOLING scope, which holds only the span's
        total and lets one entry take it all -- the ceiling must be physical or
        the reservation decides the outcome.
        """
        if budget_tokens is None:
            eval_capacity = chunk_size
            slot_capacity = 0
            slot_rows = 0
            per_group_capacity = 0
        else:
            # A pooling scope holds only its span's TOTAL, so nothing stops
            # one entry taking it all. The only non-arbitrary ceiling is the
            # physical one, so sizing for that keeps the reservation from
            # deciding the outcome. ``uniform`` keeps the tight ``budget``.
            per_group_capacity = int(max_model_len)
            if budget_scope not in POOLED_BUDGET_SCOPES:
                per_group_capacity = min(
                    int(budget_tokens), per_group_capacity)
            slot_capacity = per_group_capacity + chunk_size
            eval_capacity = (slot_capacity - n_sink_tokens
                             if evict_current_chunk
                             else per_group_capacity - n_sink_tokens)
            eval_capacity = max(eval_capacity, 0)
            # A recomputing source holds nothing between steps, so one buffer
            # does rather than one per request -- the largest term here. From
            # the knobs, the scorer being installed after this runs.
            slot_rows = (max_num_reqs
                         if slot_scores_persist_across_steps(
                             scorer, slot_score_source)
                         else 1)
        # qk scorers need float32 for a stable reduction; the gate does not.
        score_dtype = (torch.float32
                       if scorer in QK_SCORERS else model_dtype)
        return WorkspaceSpec(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            num_groups=num_groups,
            page_group_size=page_group_size,
            max_num_reqs=max_num_reqs,
            chunk_size=chunk_size,
            window_size=window_size,
            eval_capacity=eval_capacity,
            slot_capacity=slot_capacity,
            per_group_capacity=per_group_capacity,
            budget_scope=budget_scope,
            slot_rows=slot_rows,
            score_dtype=score_dtype,
        )


class CompressionWorkspace:
    """Owns every preallocated compression tensor and the row allocator."""

    def __init__(self, spec: WorkspaceSpec, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        try:
            self._allocate()
        except torch.OutOfMemoryError as exc:
            raise ValueError(
                f"The KV compression workspace does not fit on the device: "
                f"{spec.max_num_reqs} rows for eval_capacity="
                f"{spec.eval_capacity}, slot_capacity={spec.slot_capacity}, "
                f"chunk={spec.chunk_size}. Lower --max-num-seqs, "
                f"--compression-budget-tokens or --compression-chunk-size."
            ) from exc
        self._free_rows: list[int] = list(reversed(range(spec.max_num_reqs)))
        self.log_reservation()
        self._reject_if_unusable()

    def _allocate(self) -> None:
        spec = self.spec
        device = self.device
        num_layers = spec.num_layers
        num_kv_heads = spec.num_kv_heads
        num_groups = spec.num_groups
        num_rows = spec.max_num_reqs
        dtype = spec.score_dtype

        # ---- shared, step-local ------------------------------------------
        # One chunk's scores as [previous window | chunk]: the ratio regime's
        # eval region is a view into this, the budget regime only lands the
        # fresh chunk here.
        self.staging = torch.empty(
            num_layers, num_kv_heads, spec.window_size + spec.chunk_size,
            dtype=dtype, device=device)
        # MEMBER order: the budget scope maps members to clusters itself.
        self.eval_scores = torch.empty(
            num_layers, num_kv_heads, spec.eval_capacity,
            dtype=dtype, device=device)
        # The same scores in (cluster, column) order, as the writeback indexes
        # them. Sorted in place: the sorted VALUES are never read.
        self.rank_scores = torch.empty(
            num_layers, num_groups, spec.page_group_size, spec.eval_capacity,
            dtype=dtype, device=device)
        # int64 is what ``torch.sort`` writes; a narrower slab would need an
        # int64 temporary of the same shape and raise the peak instead.
        self.sorted_index = torch.empty(
            num_layers, num_groups, spec.page_group_size, spec.eval_capacity,
            dtype=torch.int64, device=device)
        # Scratch for the write-back kernel's in-place sort of the selected
        # positions: one byte per ``sorted_index`` cell.
        self.keep_mask = torch.zeros(
            num_layers, num_groups, spec.page_group_size, spec.eval_capacity,
            dtype=torch.uint8, device=device)

        # ---- per row, persists across steps ------------------------------
        # The chunk's scores so far, for a chunk the scheduler split over
        # several steps. ``pending_len`` is the per-(row, layer) write cursor.
        self.pending_score = torch.empty(
            num_rows, num_layers, num_kv_heads, spec.chunk_size,
            dtype=dtype, device=device)
        self.pending_len = np.zeros((num_rows, num_layers), dtype=np.int32)
        # The one region a later ratio-regime chunk still re-ranks.
        self.prior_window = torch.empty(
            num_rows, num_layers, num_kv_heads, spec.window_size,
            dtype=dtype, device=device)
        # Permanently kept positions, and the length the last eviction left.
        self.locked = torch.zeros(
            num_rows, num_layers, num_groups, dtype=torch.long, device=device)
        self.valid_lengths = torch.zeros(
            num_rows, num_layers, num_groups, dtype=torch.long, device=device)
        # Per-position statistics in cache-slot coordinates (budget regime).
        # ``slot_rows`` is ``num_rows`` only when the source keeps scores
        # between steps; otherwise ``stat_buffer_for`` shares one row.
        self.stat_buffer: torch.Tensor | None = (
            torch.empty(
                spec.slot_rows, num_layers, num_kv_heads, spec.slot_capacity,
                dtype=dtype, device=device)
            if spec.slot_capacity > 0 and spec.slot_rows > 0 else None)

    def stat_buffer_for(self, row: int) -> torch.Tensor:
        """This request's slice of the per-position statistics.

        A source that rewrites every live slot shares one slice with the whole
        step, its values never outliving the request loop's iteration; one that
        keeps history gets its own row.
        """
        if self.stat_buffer is None:
            raise RuntimeError(
                "stat_buffer_for: the workspace was built without per-position "
                "statistics (slot_capacity == 0), so no regime may ask for "
                "them.")
        return self.stat_buffer[row if self.spec.slot_rows > 1 else 0]

    # ------------------------------------------------------------------ rows
    def acquire_row(self) -> int:
        """Reserve a row for one request. Exhaustion means a row outlived the
        request that held it -- a bookkeeping bug, not a capacity question."""
        if not self._free_rows:
            raise RuntimeError(
                f"CompressionWorkspace: all {self.spec.max_num_reqs} rows are "
                "in use. Rows are released when a request finishes or is "
                "preempted, so one of those paths missed a request.")
        return self._free_rows.pop()

    def release_row(self, row: int) -> None:
        """Return a row to the pool and clear the state that must not leak into
        the next request to occupy it."""
        self.pending_len[row, :] = 0
        self.locked[row].zero_()
        self.valid_lengths[row].zero_()
        self._free_rows.append(row)

    # ------------------------------------------------------------- reporting
    @property
    def shared_bytes(self) -> int:
        tensors = [self.staging, self.eval_scores, self.rank_scores,
                   self.sorted_index, self.keep_mask]
        # Shared exactly when they hold nothing between steps.
        if self.stat_buffer is not None and self.spec.slot_rows <= 1:
            tensors.append(self.stat_buffer)
        return sum(t.numel() * t.element_size() for t in tensors)

    @property
    def per_row_bytes(self) -> int:
        tensors = [self.pending_score, self.prior_window,
                   self.locked, self.valid_lengths]
        if self.stat_buffer is not None and self.spec.slot_rows > 1:
            tensors.append(self.stat_buffer)
        return sum(t.numel() * t.element_size() for t in tensors)

    @property
    def reserved_bytes(self) -> int:
        return self.shared_bytes + self.per_row_bytes

    def log_reservation(self) -> None:
        """Report the reservation at startup. The profiling that follows
        subtracts it from the KV pool, so it belongs in the log where a
        too-small KV cache is diagnosed."""
        spec = self.spec
        logger.info(
            "KV compression workspace reserved %.1f MiB "
            "(shared %.1f MiB, %d rows x %.1f MiB): eval_capacity=%d, "
            "slot_capacity=%d (%d rows), chunk=%d, score dtype=%s.",
            self.reserved_bytes / _MIB,
            self.shared_bytes / _MIB,
            spec.max_num_reqs,
            self.per_row_bytes / max(spec.max_num_reqs, 1) / _MIB,
            spec.eval_capacity, spec.slot_capacity, spec.slot_rows,
            spec.chunk_size, spec.score_dtype)
        if self.reserved_bytes > 1024 * _MIB:
            logger.warning(
                "The KV compression workspace reserves %.2f GiB, which is "
                "taken out of the KV cache pool. It is dominated by the "
                "per-row term, so --max-num-seqs is the strongest knob; "
                "--compression-budget-tokens and --compression-chunk-size "
                "reduce it too.",
                self.reserved_bytes / (1024 * _MIB))

    def _reject_if_unusable(self) -> None:
        """Fail now if the reservation leaves no room for a KV cache.

        Otherwise it surfaces later as "not enough memory for the KV cache",
        which points at the wrong knob. The row pool is sized by
        ``max_num_seqs`` while the concurrency a long budget sustains is bounded
        by the KV pool instead, so the two are easy to mis-pair.
        """
        if self.device.type != "cuda":
            return
        _, total = torch.cuda.mem_get_info(self.device)
        if self.reserved_bytes <= 0.5 * total:
            return
        spec = self.spec
        raise ValueError(
            f"The KV compression workspace would reserve "
            f"{self.reserved_bytes / (1024 * _MIB):.2f} GiB of the device's "
            f"{total / (1024 * _MIB):.2f} GiB, leaving too little for the KV "
            f"cache. It is {spec.max_num_reqs} rows x "
            f"{self.per_row_bytes / max(spec.max_num_reqs, 1) / _MIB:.1f} MiB "
            f"plus {self.shared_bytes / _MIB:.1f} MiB shared. Lower "
            f"--max-num-seqs (currently {spec.max_num_reqs}) to the concurrency "
            f"you actually need — with a large --compression-budget-tokens the "
            f"KV cache itself bounds concurrency well below the default — or "
            f"lower --compression-budget-tokens / --compression-chunk-size.")
