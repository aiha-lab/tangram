# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache compression functionality mixin for the GPU model runner.

This mixin holds the Tangram prefill-with-eviction compression subsystem that
was previously inlined in ``GPUModelRunner``. It is a behaviour-preserving
extraction: the methods still run against the runner's own attributes (they are
mixed in via inheritance, so ``self`` is the ``GPUModelRunner`` instance), which
is why the compression code is kept together here rather than behind a
composed collaborator — it reads and writes the runner's live per-step state
(``input_batch``, ``kv_caches``, the ``compressor`` / ``compression_executor``
handles, and the ``pending_*`` result buffers the post-forward fold-in drains)
too pervasively for a hand-off interface to be worthwhile.

The runner owns the state these methods touch:

* ``compressor`` / ``compression_executor`` — created in ``_init_compression``
  and declared (as ``None``) in ``GPUModelRunner.__init__``; read from the
  non-compression paths too, so they stay on the runner.
* ``pending_eff_seq_lens`` / ``pending_freed_blocks`` /
  ``pending_sliding_freed_blocks`` / ``last_sliding_freed_block_ids`` — the
  per-step scratch buffers, also initialised on the runner.
* ``compression_static_layer_ids`` / ``compression_sliding_layer_ids`` /
  ``compression_sliding_window`` — set by ``_init_compression`` and read only by
  the methods in this mixin.

Whenever compression is disabled the runner never constructs a compressor, and
none of these methods run (their call sites are all guarded), so the mixin adds
zero per-step overhead to the dense path.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm.config import get_layers_from_vllm_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    full_attention_layer_indices,
    sliding_window_layers,
)
from vllm.v1.attention.compression import (
    CompressionExecutor,
    CompressionMetadata,
    KVCompressor,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import (
        CompressionRequestMetadata,
        SchedulerOutput,
    )

logger = init_logger(__name__)


# Defined as a KV-cache compression functionality mixin for GPUModelRunner.
class CompressionModelRunnerMixin:
    def _init_compression(self) -> None:
        """Build the compressor + executor and wire per-layer scorers.
        Called from ``load_model`` after the model is constructed.
        ``cache_config.num_kv_heads`` is model-global; the compressor and
        executor live per-rank.
        """
        cache_config = self.cache_config
        assert cache_config.compression_enabled
        assert cache_config.page_group_size is not None
        assert cache_config.num_kv_heads is not None
        num_layers = cache_config.num_hidden_layers
        assert num_layers is not None and num_layers > 0
        head_size = self.model_config.get_head_size()
        hidden_dim = self.model_config.get_hidden_size()
        block_size = cache_config.block_size
        dtype = self.dtype
        num_kv_heads_total = cache_config.num_kv_heads
        num_kv_heads_per_rank = self.model_config.get_num_kv_heads(
            self.parallel_config)
        tp_rank = get_tensor_model_parallel_rank()
        if num_kv_heads_per_rank % cache_config.page_group_size != 0:
            raise ValueError(
                f"Compression: per-rank num_kv_heads ({num_kv_heads_per_rank}, "
                f"from total {num_kv_heads_total} / tp_size "
                f"{self.parallel_config.tensor_parallel_size}) must be a "
                f"multiple of page_group_size ({cache_config.page_group_size})."
            )

        # Enumerate the decoder attention layers and split them into
        # full-attention (compressible) vs sliding-window. FastKVZip scores and
        # evicts only the full-attention layers; sliding-window layers retain
        # their full KV and are never compressed (the window is applied inside
        # the attention kernel). The gate checkpoint stores one gate per
        # full-attention layer. For a dense model every layer is full-attention,
        # so this reduces exactly to the prior all-layers path.
        #
        # The *outer* attention block (e.g. Qwen2Attention) exposes
        # hidden_states for the gate scorer; the inner ``Attention`` only sees
        # the already-projected q/k/v but exposes ``sliding_window`` for the
        # split.
        from vllm.attention import Attention as _InnerAttention
        from vllm.model_executor.models.utils import extract_layer_index

        inner_attn_layers = get_layers_from_vllm_config(
            self.vllm_config, _InnerAttention
        )
        # ``layer_to_parent`` feeds the hidden_states scorer (FastKVZip gate);
        # ``layer_to_inner`` feeds the query/key scorer (SnapKV), which reads
        # the inner ``Attention``'s post-RoPE q/k. Both are keyed by physical
        # layer index so ``attach_scorers`` can pick by scorer type.
        layer_to_parent: dict[int, nn.Module] = {}
        layer_to_inner: dict[int, nn.Module] = {}
        for layer_name, inner in inner_attn_layers.items():
            try:
                idx = extract_layer_index(layer_name)
            except (AssertionError, ValueError):
                # Non-decoder attention (e.g. encoder-only).
                continue
            parts = layer_name.rsplit(".", 1)
            if len(parts) != 2:
                continue
            parent_name = parts[0]
            try:
                parent = self.model.get_submodule(parent_name)
            except AttributeError:
                continue
            layer_to_parent[idx] = parent
            layer_to_inner[idx] = inner

        missing = [i for i in range(num_layers) if i not in layer_to_parent]
        if missing:
            raise RuntimeError(
                f"Compression: outer attention parent missing for layers "
                f"{missing}; one Attention per decoder layer is required."
            )

        # Physical indices of the compressible (full-attention) layers, in
        # ascending order — the order the gate checkpoint's per-layer modules
        # are stored in. Shared with the ragged FlashAttention builder via
        # full_attention_layer_indices so the two cannot disagree on which
        # layers are compressed (and thus how a cluster map maps to physical
        # rows).
        static_layer_ids = full_attention_layer_indices(self.vllm_config)
        if not static_layer_ids:
            raise RuntimeError(
                "Compression: model has no full-attention layers; FastKVZip "
                "needs at least one compressible (non-sliding-window) layer."
            )
        num_compressed_layers = len(static_layer_ids)
        # Consumed by the executor (KV / block-table access is physical) and by
        # the compression layer loop (static->physical kept_lengths expansion).
        self.compression_static_layer_ids = np.array(
            static_layer_ids, dtype=np.int64)

        # Sliding-window layers + window, for sliding-window KV eviction.
        # Out-of-window front blocks of these layers are returned to the pool at
        # each compression boundary so concurrent long-context requests stop
        # thrashing on KV (ragged paging otherwise keeps full KV for every
        # layer, including sliding-window ones). Resolved via the same source of
        # truth as the compressible (full-attention) split. Empty for a dense
        # model (no sliding-window layers) → eviction is a no-op.
        sliding_ids, sliding_window = sliding_window_layers(self.vllm_config)
        self.compression_sliding_layer_ids = np.array(
            sliding_ids, dtype=np.int64)
        self.compression_sliding_window = sliding_window

        # Sliding-window hybrid models author the cluster map over the
        # compressible (full-attention) layers only: shape
        # ``[num_compressed_layers, num_kv_heads]``. The compressor consumes it
        # as-is (its layer axis IS the compressed space; set_cluster_map below
        # validates ``cluster_of.shape[0] == num_compressed_layers``). The
        # FlashAttention builder expands the same static map to the physical
        # layer layout it needs (static cross-layer clusters at their executor
        # block-table rows + identity for sliding layers) — see
        # ragged_layout.physical_member_maps_from_static_cluster_map.
        # Dense models keep ``num_compressed_layers == num_layers`` and the map
        # is already physical, so both paths see the same array.

        self.compressor = KVCompressor(
            num_layers=num_compressed_layers,
            num_kv_heads=num_kv_heads_per_rank,
            page_group_size=cache_config.page_group_size,
            head_size=head_size,
            hidden_dim=hidden_dim,
            block_size=block_size,
            dtype=dtype,
            device=self.device,
            level=cache_config.compression_level,
        )
        # Axis-2 scorer selection. FastKVZip loads a
        # per-layer gate checkpoint over hidden_states; every other scorer is a
        # gate-free query/key scorer (SnapKV, KeyDiff, …) dispatched by name
        # through ``build_qk_scorer`` (num_q_per_kv = per-rank GQA ratio).
        if cache_config.compression_scorer == "fastkvzip":
            self.compressor.load_gate_checkpoint(
                self.model_config.model,
                cache_config.compression_gate_path,
                num_kv_heads_total=num_kv_heads_total,
                tp_rank=tp_rank,
            )
        else:
            self.compressor.set_qk_scorers(
                cache_config.compression_scorer,
                num_q_per_kv=self.model_config.get_num_attention_heads(
                    self.parallel_config) // num_kv_heads_per_rank,
                snap_window=cache_config.compression_snap_window,
                snap_kernel=cache_config.compression_snap_kernel,
                ea_use_covariance=cache_config.compression_ea_use_covariance,
                ea_use_vnorm=cache_config.compression_ea_use_vnorm,
                ea_n_future_positions=(
                    cache_config.compression_ea_n_future_positions),
                ea_epsilon=cache_config.compression_ea_epsilon,
            )
        # Bind the same member->cluster map the FlashAttention builder uses so
        # scoring max-pools over the physical clusters (cross-layer when a map
        # is set; identity adjacency otherwise — bit-identical to before).
        self.compressor.set_cluster_map(cache_config.head_group_cluster_map)

        # Offline retention profiling (head-group clustering): when a dump
        # directory is configured, observe every keep decision. None in
        # production, so the keep-decision path stays untouched.
        if cache_config.compression_retention_dump is not None:
            from vllm.v1.attention.compression.profiling import (
                RetentionProfileObserver)
            logger.warning(
                "compression_retention_dump is set to '%s': attaching the "
                "offline retention profiler. This writes a dump file per keep "
                "decision and adds overhead — leave it unset in production.",
                cache_config.compression_retention_dump)
            # Tag dumps with the TP rank: each rank observes only its own KV-head
            # shard, and all ranks share the dump directory.
            self.compressor.keep_decision_observer = RetentionProfileObserver(
                cache_config.compression_retention_dump,
                rank=get_tensor_model_parallel_rank())

        # Attach the scorers to the compressible layers in ascending
        # physical-layer order (matching the scorer ordering and the
        # compressor's per-compressed-layer state). The hidden_states scorer
        # is delivered through the outer block; the query/key scorer through
        # the inner Attention — attach_scorers picks per scorer type.
        static_parents = [layer_to_parent[i] for i in static_layer_ids]
        static_inners = [layer_to_inner[i] for i in static_layer_ids]
        self.compressor.attach_scorers(static_parents, static_inners)

        self.compression_executor = CompressionExecutor(
            num_layers=num_layers,
            num_kv_heads_per_layer=num_kv_heads_per_rank,
            page_group_size=cache_config.page_group_size,
            head_size=head_size,
            block_size=block_size,
            compressed_layer_ids=static_layer_ids,
        )

    def _begin_compression_step(
        self,
        scheduler_output: "SchedulerOutput",
        compression_metadata: dict[str, "CompressionRequestMetadata"],
    ) -> None:
        """Activate the compressor for this step. Computes each
        compression-active request's token range in the upcoming forward's
        ``hidden_states`` (prefix-sum over scheduled token counts) and
        stashes them as ``pending_req_offsets`` for the per-layer scorers.
        """
        assert self.compressor is not None
        if not compression_metadata:
            return

        req_ids = self.input_batch.req_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        offsets: list[tuple[str, int, int]] = []
        # ``req_id -> global position of this chunk's first scored token`` (the
        # request's num_computed_tokens this step). Position-dependent qk
        # scorers (StreamingLLM, ExpectedAttention) read it in the query/key
        # scorer.
        pos_offsets: dict[str, int] = {}
        cursor = 0
        for req_index, req_id in enumerate(req_ids):
            num_tokens = num_scheduled_tokens.get(req_id, 0)
            if num_tokens <= 0:
                continue
            start = cursor
            end = cursor + num_tokens
            cursor = end
            if req_id in compression_metadata:
                offsets.append((req_id, start, end))
                pos_offsets[req_id] = int(
                    self.input_batch.num_computed_tokens_cpu[req_index])

        # First-chunk resets happen earlier in
        # ``_pre_prepare_compression_reset``; this only adds fresh state.
        for req_id in compression_metadata:
            if req_id not in self.compressor.req_state:
                self.compressor.begin_request(req_id)

        self.compressor.compress_active = True
        self.compressor.pending_req_offsets = offsets
        self.compressor.pending_req_pos_offsets = pos_offsets

    def _pre_prepare_compression_reset(
        self,
        compression_metadata: dict[str, "CompressionRequestMetadata"],
    ) -> None:
        """Reset stale per-request state on the first chunk before
        slot_mapping and attention metadata are built. On preempt-resume
        the worker's ``effective_seq_lens_cpu`` and compressor
        ``req_state`` carry pre-preempt accumulators; clearing here keeps
        slot_mapping inside the fresh block range and prevents stale
        scores from leaking into the cross-layer threshold. Idempotent
        for fresh requests.
        """
        if not compression_metadata or self.compressor is None:
            return
        effective_seq_lens_cpu = self.input_batch.effective_seq_lens_cpu
        for req_id, md in compression_metadata.items():
            if md.chunk_in_sequence_idx != 0:
                continue
            if req_id in self.compressor.req_state:
                self.compressor.end_request(req_id)
            self.compressor.begin_request(req_id)
            row = (self.input_batch.req_id_to_index.get(req_id)
                   if effective_seq_lens_cpu is not None else None)
            if effective_seq_lens_cpu is not None and row is not None:
                effective_seq_lens_cpu[row, :] = 0

    def _prev_kept_lengths(
        self, req_id: str, num_static: int, num_groups: int
    ) -> np.ndarray:
        """Per-(compressible-layer, group) valid length as of the previous
        compression boundary.

        This is the authoritative pre-chunk length the keep arithmetic needs —
        the ``kept_lengths`` the last eviction left, not the live
        ``effective_seq_lens``. Sourcing it from compressor state is what lets
        one compression chunk span multiple forward steps: during budget-sliced
        sub-chunks ``effective_seq_lens`` holds the raw write extent (kept +
        accumulated), while the decision needs the pre-chunk length.

        ``cached_kept_lengths_cpu`` carries that value (already on the CPU and
        cross-rank-MAX-reduced under tensor parallelism). It is set at the last
        boundary and persists across sub-chunks: ``prepare_keep_decision`` clears
        it only at its own end, which runs after this read. On the first chunk it
        is ``None`` and we return zeros, matching the post-reset
        ``effective_seq_lens``. In the common single-step chunk it equals
        ``eff_phys[static_layer_ids]``, so the decision is byte-identical to the
        pre-change path.

        Returns ``[num_static, num_groups]`` int64.
        """
        cached_kept = self.compressor.req_state[req_id].cached_kept_lengths_cpu
        if cached_kept is not None:
            return cached_kept.astype(np.int64, copy=True)
        return np.zeros((num_static, num_groups), dtype=np.int64)

    def _expand_static_to_physical_lengths(
        self,
        eff_phys: np.ndarray,
        this_step: int,
        kept_lengths_static: np.ndarray,
        static_layer_ids: np.ndarray,
    ) -> np.ndarray:
        """Rebuild the physical (all-layer) kept_lengths from the compressible
        ones.

        Compressible (full-attention) layers take their post-eviction lengths;
        sliding-window layers keep their full KV and so grow by ``this_step``
        (this forward's raw advance) on top of ``eff_phys`` (the pre-increment
        value). ``this_step`` — not ``chunk_len`` — is correct because earlier
        sub-chunks of the same compression chunk were already folded into
        ``eff_phys``. For a dense model every layer is compressible, so the
        static assignment overwrites the whole array.

        Returns ``[num_layers, num_groups]`` int32.
        """
        kept_lengths_phys = (eff_phys + this_step).astype(np.int32)
        kept_lengths_phys[static_layer_ids] = kept_lengths_static
        return kept_lengths_phys

    def _evict_sliding_window_blocks(
        self,
        block_table,
        row_idx: int,
        kept_lengths_phys: np.ndarray,
    ) -> None:
        """Free the sliding-window layers' out-of-window front KV blocks.

        Under ragged paging the sliding-window layers keep full KV (they
        are not compressed), but FlashAttention only attends to the last
        ``sliding_window`` tokens, so the leading blocks outside the window are
        dead weight. This returns them to the pool null-in-place — the in-window
        tail keeps its block positions, so output is unchanged — and records the
        freed ids on ``pending_sliding_freed_blocks`` for the scheduler's
        ``null_blocks_by_ids`` (length-preserving).

        No-op for a dense model (no sliding-window layers). The sliding length
        is the full per-layer sequence length, uniform across the sliding-window
        layers, so a single skip count covers all of them.
        """
        sliding_layer_ids = self.compression_sliding_layer_ids
        if not sliding_layer_ids.size:
            return
        block_size = self.compression_executor.block_size
        sliding_len = int(kept_lengths_phys[sliding_layer_ids[0], 0])
        num_skipped_blocks = (
            sliding_len - self.compression_sliding_window + 1) // block_size
        if num_skipped_blocks <= 0:
            return
        freed = block_table.null_front_blocks_sliding(
            row_idx=row_idx,
            sliding_layer_ids=sliding_layer_ids,
            num_head_groups_per_layer=(
                self.compression_executor.num_head_groups_per_layer),
            num_skipped_blocks=num_skipped_blocks,
        )
        if freed.size:
            self.pending_sliding_freed_blocks.append(freed)

    def _run_compression_layer_loop(
        self,
        compression_metadata: dict[str, "CompressionRequestMetadata"],
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Drive ``executor.run_request`` once per request that closed a
        compression-chunk boundary this step (``run_compression`` is True;
        the caller passes only those). Each request evicts over its
        accumulated ``compression_chunk_len`` and writes its updates into
        ``pending_*`` for the post-forward fold-in. Budget-sliced sub-chunk
        steps (``run_compression`` False) never reach here — their KV is
        written raw and their gate scores accumulate in the compressor.
        """
        assert self.compressor is not None
        assert self.compression_executor is not None
        block_table = self.input_batch.block_table.block_tables[0]
        eff_seq_lens_cpu = self.input_batch.effective_seq_lens_cpu
        num_groups = self.compression_executor.num_head_groups_per_layer
        num_layers = self.compression_executor.num_layers  # physical (all)
        # Terminology used below: "physical" layers are all layers (the KV cache
        # and block table span every one). "static" == "compressible" == the
        # full-attention layers — the only ones the compressor scores and evicts
        # (sliding-window layers keep their full KV). ``static_layer_ids`` holds
        # their physical indices, so it maps a compressed position to its
        # physical layer. For a dense model every layer is compressible and the
        # static<->physical selects / expands below are identities.
        static_layer_ids = self.compression_static_layer_ids

        tp_world_size = get_tensor_model_parallel_world_size()
        tp_group = get_tp_group() if tp_world_size > 1 else None
        tp_device = self.device if tp_world_size > 1 else None

        for req_id, req_md in compression_metadata.items():
            row_idx = self.input_batch.req_id_to_index[req_id]
            # ``this_step`` advances the (uncompressed) sliding-layer lengths;
            # ``chunk_len`` is the accumulated span the eviction evaluates
            # (== chunk_size for interior chunks). They differ only when budget
            # sharing split this chunk across forward steps.
            this_step = scheduler_output.num_scheduled_tokens[req_id]
            chunk_len = req_md.compression_chunk_len
            metadata = CompressionMetadata(
                req_id=req_id,
                row_idx=row_idx,
                chunk_len=chunk_len,
                floor_min=req_md.floor_min,
            )

            # The compressor scores/evicts only the compressible layers, so it
            # is fed the per-(compressible-layer, group) lengths; sliding-window
            # layers keep their full KV and are excluded. ``eff_phys`` is the
            # physical (all-layer) view used to rebuild the sliding lengths.
            eff_phys = (
                eff_seq_lens_cpu[row_idx, :]
                .astype(np.int64, copy=True)
                .reshape(num_layers, num_groups)
            )
            num_static = len(static_layer_ids)
            prev_seq_lens_static = self._prev_kept_lengths(
                req_id, num_static, num_groups)

            # Cross-layer KeepDecision; caches sorted indices + group scores
            # for ``run_request`` (indexed by compressible position).
            self.compressor.prepare_keep_decision(
                req_id=req_id,
                prev_seq_lens_per_layer=torch.from_numpy(
                    prev_seq_lens_static),
                chunk_len=chunk_len,
                ratio=req_md.compression_ratio,
                window_size=req_md.window_size,
                n_sink_tokens=req_md.n_sink_tokens,
                total_prompt_tokens=req_md.total_prompt_tokens,
            )

            # Per-compressible-layer post-evict kept_lengths. Under TP,
            # MAX-reduce across ranks so every worker frees the same block ids
            # (the slot is shared across ranks for a given
            # (req, layer, group_local)). No-op at TP=1.
            kept_lengths_static = (
                self.compressor.compute_kept_lengths_per_rank(
                    req_id=req_id,
                    eff_seq_lens_row=prev_seq_lens_static.reshape(-1),
                    chunk_len=chunk_len,
                    floor_min=req_md.floor_min,
                )
            )
            if tp_world_size > 1:
                kept_lengths_gpu = torch.from_numpy(
                    kept_lengths_static).to(tp_device)
                torch.distributed.all_reduce(
                    kept_lengths_gpu,
                    op=torch.distributed.ReduceOp.MAX,
                    group=tp_group.device_group,
                )
                kept_lengths_static = (
                    kept_lengths_gpu.cpu().numpy().astype(np.int32))
                self.compressor.req_state[
                    req_id].cached_kept_lengths_cpu = kept_lengths_static

            self.compression_executor.run_request(
                layer_kv_caches=self.kv_caches,
                block_table=block_table,
                prev_seq_lens_static_cpu=prev_seq_lens_static,
                compressor=self.compressor,
                compression_metadata=metadata,
            )

            kept_lengths_phys = self._expand_static_to_physical_lengths(
                eff_phys, this_step, kept_lengths_static, static_layer_ids)
            self.pending_eff_seq_lens[req_id] = (
                kept_lengths_phys.reshape(-1))

            # Batched compact: one numpy scan over all (layer, group)
            # pairs instead of one Python call per layer.
            block_size = self.compression_executor.block_size
            new_num_blocks_per_layer = (
                (kept_lengths_phys + block_size - 1) // block_size
            ).astype(np.int32)
            freed = block_table.compact_after_compress_all_layers(
                row_idx=row_idx,
                num_head_groups_per_layer=num_groups,
                new_num_blocks_per_layer=new_num_blocks_per_layer,
            )
            if freed.size:
                self.pending_freed_blocks.append(freed)

            # Sliding-window eviction: free the sliding-window layers'
            # out-of-window front blocks (kept full by compression but never
            # attended past the window). No-op for dense models.
            self._evict_sliding_window_blocks(
                block_table, row_idx, kept_lengths_phys)

    def _postprocess_compress_updates(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Fold pending compression results into the input batch.

        eff_seq_lens becomes the next step's source of truth; freed
        block ids are returned for the scheduler to release.
        """
        for req_id, eff_lens in self.pending_eff_seq_lens.items():
            row = self.input_batch.req_id_to_index.get(req_id)
            if row is None:
                # Request was removed in the same step — drop silently;
                # its KV is already on its way back to the pool.
                continue
            self.input_batch.effective_seq_lens_cpu[row, :] = eff_lens

        new_eff_seq_lens = dict(self.pending_eff_seq_lens)
        if self.pending_freed_blocks:
            # ``np.unique`` sorts + deduplicates in one C-level pass —
            # the dedup keeps the post-condition of the old set-based
            # path (no id reported twice to ``free_blocks_by_ids``).
            freed_block_ids = np.unique(
                np.concatenate(self.pending_freed_blocks)
            )
        else:
            freed_block_ids = np.empty(0, dtype=np.int32)
        # Sliding-window freed ids travel on their own channel (null-in-place,
        # not the shrinking compression free) — see ``null_blocks_by_ids``.
        if self.pending_sliding_freed_blocks:
            self.last_sliding_freed_block_ids = np.unique(
                np.concatenate(self.pending_sliding_freed_blocks)
            )
        else:
            self.last_sliding_freed_block_ids = np.empty(0, dtype=np.int32)
        self.pending_eff_seq_lens.clear()
        self.pending_freed_blocks.clear()
        self.pending_sliding_freed_blocks.clear()
        return new_eff_seq_lens, freed_block_ids

    def _end_compression_step(self) -> None:
        """Clear the compress-active flag."""
        if self.compressor is None:
            return
        self.compressor.compress_active = False
        self.compressor.pending_req_offsets = None
        self.compressor.pending_req_pos_offsets = None

    @contextmanager
    def _compression_step(self, scheduler_output, compression_metadata):
        """Compression-active context for one step. Yields True when
        compression runs this step (caller should then invoke the
        post-forward loop), False otherwise. ``_end_compression_step``
        always fires on exit, even on exception.
        """
        active = bool(compression_metadata)
        if active:
            assert self.compressor is not None, (
                "scheduler emitted compression_metadata but the runner has "
                "no KVCompressor; compression_ratio < 1.0 must be set."
            )
            self._begin_compression_step(scheduler_output, compression_metadata)
        try:
            yield active
        finally:
            if active:
                self._end_compression_step()

    def _build_effective_seq_lens_increments(
        self,
        num_scheduled_tokens: dict,
        exclude: dict | None = None,
    ) -> np.ndarray | None:
        """Per-row increments for the post-forward
        ``effective_seq_lens_cpu`` update. Returns ``None`` when no row
        is active.
        """
        num_reqs = self.input_batch.num_reqs
        if num_reqs == 0:
            return None
        tokens_per_row = np.zeros(num_reqs, dtype=np.int32)
        req_id_to_index = self.input_batch.req_id_to_index
        any_active = False
        for req_id, num_tokens in num_scheduled_tokens.items():
            if num_tokens <= 0:
                continue
            if exclude is not None and req_id in exclude:
                continue
            row = req_id_to_index.get(req_id)
            if row is None:
                continue
            tokens_per_row[row] = num_tokens
            any_active = True
        return tokens_per_row if any_active else None

