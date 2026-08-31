# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preallocated GPU memory for the KV-compression keep decision.

Every tensor the keep decision needs is allocated ONCE here, from sizes that
follow from the configuration alone, and is then reused for the lifetime of the
engine. Nothing in the compression path allocates a decision-sized tensor while
requests are in flight. This matters for two reasons:

* The worker profiles peak memory AFTER ``load_model`` and sizes the KV cache
  pool with whatever is left (``GPUWorker.determine_available_memory``). This
  workspace is built inside ``load_model``, so the profile SEES it and the pool
  shrinks accordingly. A budget or concurrency too large for the device then
  fails at startup, with the reservation printed, instead of surviving startup
  and hitting an out-of-memory error mid-generation.
* Growing per-request buffers on demand would scale with how many requests
  happen to be mid-prefill — a quantity no startup check can bound. Splitting
  the memory into "shared, reused within a step" and "one row per concurrent
  request" makes both parts exactly bounded (see below).

Two lifetimes, and the distinction is the whole design:

* **Shared, step-local.** The eval-region scores and the position ranking are
  produced by ``prepare_keep_decision`` and consumed by the executor's writeback
  within the SAME iteration of the runner's per-request loop. So one copy serves
  every request, no matter how many close a compression boundary in one step,
  and this part does NOT scale with ``max_num_seqs``.
* **Per row, persists across steps.** Scores accumulated across a chunk that the
  scheduler split over several forward steps, the per-(layer, group) lengths, and
  (budget regime only) the per-position statistics buffer must survive between
  steps, so each active request holds one row. This part scales with
  ``max_num_seqs`` and is the term to watch when sizing a run.

Rows are handed out by an allocator private to this class rather than being
indexed by the ``InputBatch`` row: the input batch compacts its rows when a
request finishes, which would silently reassign another request's buffers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
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

    ``eval_capacity`` and ``slot_capacity`` are the WIDEST values the active
    eviction regime can produce; the regime's own geometry is checked against
    them at runtime, so a shape that would have overflowed raises instead of
    silently reallocating.
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
    #: Rows of per-position statistics to reserve: ``max_num_reqs`` when the
    #: score source keeps a slot's score between steps, 1 when it rewrites every
    #: live slot at each eviction (then one buffer serves the whole step, like
    #: the shared tensors above), 0 when there are no statistics at all.
    slot_rows: int
    #: dtype the active scorer produces. Held exactly, not coerced: the gate
    #: (FastKVZip) emits the model dtype while the query/key scorers emit
    #: float32, and rounding either way would change the keep decision.
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
    ) -> "WorkspaceSpec":
        """Derive the shapes from the cache configuration.

        The bounds below are the ones the regimes guarantee
        (``eviction_regime.py``); each is asserted at runtime rather than
        trusted.

        Ratio regime — the eval region is the previous chunk's window plus the
        fresh chunk, addressed in a ``window + chunk`` staging buffer, so it can
        never exceed one chunk. No per-position statistics are kept.

        Budget regime — the keep decision caps every (layer, group) at
        ``budget``, so the live length before the next eviction is at most
        ``budget + chunk``. A budget at or above ``max_model_len`` can never
        trigger eviction, so the live length is bounded by the model length
        instead and the capacity is clamped to it. The eval region is the live
        cache minus the sink and the protected tail: with the fresh chunk
        protected the two chunk terms cancel and it is at most
        ``budget - sink``; with only the recent window protected the chunk stays
        in and it is at most ``budget + chunk - sink``.
        """
        if budget_tokens is None:
            eval_capacity = chunk_size
            slot_capacity = 0
            slot_rows = 0
        else:
            effective_budget = min(int(budget_tokens), int(max_model_len))
            slot_capacity = effective_budget + chunk_size
            eval_capacity = (slot_capacity - n_sink_tokens
                             if evict_current_chunk
                             else effective_budget - n_sink_tokens)
            eval_capacity = max(eval_capacity, 0)
            # A source that recomputes every live slot's score at each eviction
            # holds nothing between steps, so it needs one buffer and not one
            # per concurrent request — the difference is the largest term in the
            # whole reservation. Decided from the two knobs because the scorer
            # is installed after this runs (see slot_scores.py).
            slot_rows = (max_num_reqs
                         if slot_scores_persist_across_steps(
                             scorer, slot_score_source)
                         else 1)
        # Query/key scorers promote to float32 for a stable reduction; the
        # checkpoint-backed gate scores in the model dtype.
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
        # Staging for one chunk's scores, laid out [previous window | chunk].
        # The ratio regime's eval region is a view into it; the budget regime
        # uses it only as the landing area for the fresh chunk.
        self.staging = torch.empty(
            num_layers, num_kv_heads, spec.window_size + spec.chunk_size,
            dtype=dtype, device=device)
        # Scores of the eval region in MEMBER order — what the budget scope
        # consumes (it maps members to clusters itself).
        self.eval_scores = torch.empty(
            num_layers, num_kv_heads, spec.eval_capacity,
            dtype=dtype, device=device)
        # The same scores in (cluster, column) order, which is the order the
        # executor's writeback indexes. Sorting happens in place here (the
        # sorted VALUES are never read), so no separate values buffer exists.
        self.rank_scores = torch.empty(
            num_layers, num_groups, spec.page_group_size, spec.eval_capacity,
            dtype=dtype, device=device)
        # Descending position ranking per (cluster, column). int64 because that
        # is the index dtype ``torch.sort`` writes; a narrower slab would need
        # an int64 temporary of the same shape and raise the peak instead of
        # lowering it.
        self.sorted_index = torch.empty(
            num_layers, num_groups, spec.page_group_size, spec.eval_capacity,
            dtype=torch.int64, device=device)

        # ---- per row, persists across steps ------------------------------
        # Scores of the chunk so far, for a chunk the scheduler split across
        # forward steps. ``pending_len`` is the per-(row, layer) write cursor.
        self.pending_score = torch.empty(
            num_rows, num_layers, num_kv_heads, spec.chunk_size,
            dtype=dtype, device=device)
        self.pending_len = np.zeros((num_rows, num_layers), dtype=np.int32)
        # Previous chunk's window scores, the one region a later chunk of the
        # ratio regime still re-ranks.
        self.prior_window = torch.empty(
            num_rows, num_layers, num_kv_heads, spec.window_size,
            dtype=dtype, device=device)
        # Positions promoted to permanently kept, and the length the last
        # eviction left, per (layer, group).
        self.locked = torch.zeros(
            num_rows, num_layers, num_groups, dtype=torch.long, device=device)
        self.valid_lengths = torch.zeros(
            num_rows, num_layers, num_groups, dtype=torch.long, device=device)
        # Per-position statistics in cache-slot coordinates (budget regime).
        # ``slot_rows`` is ``num_rows`` only when the score source keeps a
        # slot's score between steps; otherwise one buffer is shared, and
        # ``stat_buffer_for`` hands every request the same row.
        self.stat_buffer: torch.Tensor | None = (
            torch.empty(
                spec.slot_rows, num_layers, num_kv_heads, spec.slot_capacity,
                dtype=dtype, device=device)
            if spec.slot_capacity > 0 and spec.slot_rows > 0 else None)

    def stat_buffer_for(self, row: int) -> torch.Tensor:
        """This request's slice of the per-position statistics.

        A source that rewrites every live slot at each eviction shares one
        slice with every other request in the step (the values never outlive the
        request loop's iteration); one that keeps history gets its own row.
        """
        if self.stat_buffer is None:
            raise RuntimeError(
                "stat_buffer_for: the workspace was built without per-position "
                "statistics (slot_capacity == 0), so no regime may ask for "
                "them.")
        return self.stat_buffer[row if self.spec.slot_rows > 1 else 0]

    # ------------------------------------------------------------------ rows
    def acquire_row(self) -> int:
        """Reserve a row for one request. Raises when the pool is exhausted,
        which means a row outlived the request that held it — a bookkeeping
        bug, not a capacity question."""
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
                   self.sorted_index]
        # The statistics are shared exactly when they hold nothing between
        # steps, so they are reported under the lifetime they actually have.
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
        """Report the reservation at startup. It is subtracted from the KV cache
        pool by the memory profiling that follows, so it belongs in the log
        where a too-small KV cache is diagnosed."""
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

        The reservation is subtracted from the KV pool by the profiling that
        follows, so an over-large one surfaces later as "not enough memory for
        the KV cache" — a message that points at the wrong knob. The row pool is
        sized by ``max_num_seqs``, while the concurrency a long budget can
        actually sustain is limited by the KV pool instead, so the two are easy
        to mis-pair; say so here, with the numbers.
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
