# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache compression for the GPU model runner.

Mixed in by inheritance, so ``self`` IS the ``GPUModelRunner``. These methods
read and write the runner's live per-step state -- ``input_batch``,
``kv_caches``, the ``compressor`` / ``compression_executor`` handles, the
``pending_*`` buffers the post-forward fold-in drains, and the static / sliding
layer id lists -- too pervasively for a composed collaborator to be worth the
hand-off interface.

With compression disabled the runner never builds a compressor and every call
site here is guarded, so the dense path pays nothing.
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
from vllm.v1.attention.backends.layer_split import split_attention_layers
from vllm.v1.attention.compression import (
    ChunkParams,
    CompressionExecutor,
    CompressionMetadata,
    KVCompressor,
)
from vllm.v1.attention.compression.eviction_writeback import (
    TritonWriteback,
    make_kept_kv_writeback,
)
from vllm.v1.attention.compression.slot_scores import KVCacheView
from vllm.v1.attention.compression.workspace import (
    CompressionWorkspace,
    WorkspaceSpec,
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

        # Only full-attention layers are compressible: a sliding layer keeps
        # full KV, its window applied in the kernel. One gate per such layer.
        from vllm.attention import Attention as _InnerAttention
        from vllm.model_executor.models.utils import extract_layer_index

        inner_attn_layers = get_layers_from_vllm_config(
            self.vllm_config, _InnerAttention
        )
        # The outer block has hidden_states for a gate scorer, the inner
        # ``Attention`` post-RoPE q/k for a qk scorer; both keyed physically.
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

        # Ascending, the order the gate checkpoint stores its modules in.
        # Shared with the ragged builder through ``split_attention_layers``, so
        # the two cannot disagree about which layers are compressed.
        layers = split_attention_layers(self.vllm_config)
        static_layer_ids = layers.full
        if not static_layer_ids:
            raise RuntimeError(
                "Compression: model has no full-attention layers; FastKVZip "
                "needs at least one compressible (non-sliding-window) layer."
            )
        num_compressed_layers = len(static_layer_ids)
        # The executor and the layer loop both address KV physically.
        self.compression_static_layer_ids = np.array(
            static_layer_ids, dtype=np.int64)

        # The sliding layers' out-of-window front blocks go back to the pool
        # at every boundary, or concurrent long-context requests thrash: ragged
        # paging otherwise holds full KV for every layer. Empty when dense.
        self.compression_sliding_layer_ids = np.array(
            layers.sliding, dtype=np.int64)
        self.compression_sliding_window = layers.sliding_window

        # A hybrid's cluster map is authored over compressible layers only.
        # The compressor consumes it as-is, its layer axis BEING that space;
        # the ragged builder expands the same array to physical layers.

        # Must happen BEFORE the worker profiles peak memory, which it does
        # right after ``load_model``, so the KV pool is sized around it and an
        # over-large budget fails at startup rather than mid-generation.
        workspace = CompressionWorkspace(
            WorkspaceSpec.from_cache_config(
                cache_config,
                num_layers=num_compressed_layers,
                num_kv_heads=num_kv_heads_per_rank,
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.model_config.max_model_len,
                model_dtype=dtype,
            ),
            self.device)

        self.compressor = KVCompressor(
            num_layers=num_compressed_layers,
            num_kv_heads=num_kv_heads_per_rank,
            page_group_size=cache_config.page_group_size,
            head_size=head_size,
            hidden_dim=hidden_dim,
            block_size=block_size,
            dtype=dtype,
            device=self.device,
            workspace=workspace,
            budget_scope=cache_config.compression_budget_scope,
            regime=cache_config.compression_regime,
            slot_score_source=cache_config.compression_slot_score_source,
        )
        # FastKVZip loads a per-layer gate checkpoint over hidden_states;
        # every other scorer is gate-free and dispatched by name.
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
                options=cache_config.resolved_scorer_options,
            )
        # The map the ragged builder pages with, so scoring max-pools right.
        self.compressor.set_cluster_map(cache_config.head_group_cluster_map)

        if cache_config.compression_retention_dump is not None:
            from vllm.v1.attention.compression.profiling import (
                RetentionProfileObserver)
            logger.warning(
                "compression_retention_dump is set to '%s': attaching the "
                "offline retention profiler. This writes a dump file per keep "
                "decision and adds overhead — leave it unset in production.",
                cache_config.compression_retention_dump)
            # Each rank observes its own shard into a shared directory.
            self.compressor.keep_decision_observer = RetentionProfileObserver(
                cache_config.compression_retention_dump,
                rank=get_tensor_model_parallel_rank())

        # Ascending, matching the scorer and per-layer state ordering.
        static_parents = [layer_to_parent[i] for i in static_layer_ids]
        static_inners = [layer_to_inner[i] for i in static_layer_ids]
        self.compressor.attach_scorers(static_parents, static_inners)

        page_group_size = cache_config.page_group_size
        writeback = make_kept_kv_writeback(
            device=self.device,
            block_size=block_size,
            page_group_size=page_group_size,
            head_size=head_size,
            keep_mask=self.compressor.workspace.keep_mask.view(
                -1, page_group_size,
                self.compressor.workspace.spec.eval_capacity),
        )
        if isinstance(writeback, TritonWriteback):
            # Compile before the KV pool exists, on a stand-in with the real
            # dtype and page geometry, so no request pays the JIT.
            writeback.warmup(torch.empty(
                2, 2, page_group_size, block_size, head_size,
                dtype=dtype, device=self.device))
        self.compression_executor = CompressionExecutor(
            num_layers=num_layers,
            num_kv_heads_per_layer=num_kv_heads_per_rank,
            page_group_size=page_group_size,
            head_size=head_size,
            block_size=block_size,
            compressed_layer_ids=static_layer_ids,
            writeback=writeback,
        )

    def _begin_compression_step(
        self,
        scheduler_output: "SchedulerOutput",
        compression_metadata: dict[str, "CompressionRequestMetadata"],
    ) -> None:
        """Activate the compressor for this step: prefix-sum the scheduled
        token counts into each compression-active request's range in the coming
        forward's ``hidden_states``, for the per-layer scorers to slice.
        """
        assert self.compressor is not None
        if not compression_metadata:
            return

        req_ids = self.input_batch.req_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        offsets: list[tuple[str, int, int]] = []
        # Global position of each chunk's first scored token.
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

        # First-chunk resets already happened; only add fresh state here.
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
        """Clear stale per-request state on the first chunk, before
        slot_mapping and attention metadata are built.

        A preempt-resume re-enters with ``effective_seq_lens_cpu`` and
        ``req_state`` still holding pre-preempt accumulators. Clearing here
        keeps slot_mapping inside the fresh block range and stops stale scores
        reaching the threshold. Idempotent for a fresh request.
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

        The keep arithmetic needs the ``kept_lengths`` the last eviction left,
        NOT the live ``effective_seq_lens``, which during budget-sliced
        sub-chunks holds the raw write extent. Taking it from compressor state
        is what lets one compression chunk span several forward steps.

        ``cached_kept_lengths_cpu`` holds it, already MAX-reduced across ranks
        under TP. It persists over sub-chunks, ``prepare_keep_decision``
        clearing it only after this read, and is ``None`` on the first chunk
        where zeros match the post-reset lengths.
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

        Compressible layers take their post-eviction lengths; sliding layers
        keep full KV and grow by ``this_step`` on top of the pre-increment
        ``eff_phys`` -- ``this_step`` and not ``chunk_len``, since earlier
        sub-chunks are already folded in. Returns ``[num_layers, num_groups]``.
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

        Ragged paging keeps these layers' full KV, but the kernel attends only
        to the last ``sliding_window`` tokens, so the leading blocks are dead
        weight. Freed null-in-place -- the in-window tail keeps its block
        positions, so output is unchanged -- and recorded for the scheduler.
        No-op for a dense model, and one skip count covers every sliding layer
        because their length is uniform.
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
        compression-chunk boundary this step -- the caller passes only those.
        Each evicts over its accumulated ``compression_chunk_len`` and writes
        its updates into ``pending_*`` for the post-forward fold-in. A
        budget-sliced sub-chunk step never reaches here: its KV is written raw
        and its scores accumulate in the compressor.
        """
        assert self.compressor is not None
        assert self.compression_executor is not None
        block_table = self.input_batch.block_table.block_tables[0]
        eff_seq_lens_cpu = self.input_batch.effective_seq_lens_cpu
        num_groups = self.compression_executor.num_head_groups_per_layer
        num_layers = self.compression_executor.num_layers  # physical (all)
        # Compressed position -> physical layer; identity for a dense model.
        static_layer_ids = self.compression_static_layer_ids

        tp_world_size = get_tensor_model_parallel_world_size()
        tp_group = get_tp_group() if tp_world_size > 1 else None
        tp_device = self.device if tp_world_size > 1 else None

        for req_id, req_md in compression_metadata.items():
            row_idx = self.input_batch.req_id_to_index[req_id]
            # ``this_step`` advances the sliding lengths, ``chunk_len`` is the
            # span the eviction evaluates; they differ on a split chunk.
            this_step = scheduler_output.num_scheduled_tokens[req_id]
            chunk_len = req_md.compression_chunk_len
            metadata = CompressionMetadata(
                req_id=req_id,
                row_idx=row_idx,
                chunk_len=chunk_len,
                floor_min=req_md.floor_min,
            )

            # The physical all-layer view, used to rebuild the sliding lengths.
            eff_phys = (
                eff_seq_lens_cpu[row_idx, :]
                .astype(np.int64, copy=True)
                .reshape(num_layers, num_groups)
            )
            num_static = len(static_layer_ids)
            prev_seq_lens_static = self._prev_kept_lengths(
                req_id, num_static, num_groups)

            # Read access to the cached keys, for a score relative to them and
            # so recomputed each eviction rather than stored.
            cache_view = KVCacheView(
                layer_kv_caches=self.kv_caches,
                block_table_gpu=block_table.block_table.gpu,
                row_idx=row_idx,
                compressed_layer_ids=static_layer_ids,
                num_groups=num_groups,
                block_size=self.compression_executor.block_size,
            )
            self.compressor.prepare_keep_decision(
                req_id=req_id,
                prev_seq_lens_per_layer=torch.from_numpy(
                    prev_seq_lens_static),
                chunk_len=chunk_len,
                cache_view=cache_view,
                params=ChunkParams(
                    keep_ratio=req_md.compression_keep_ratio,
                    budget_tokens=req_md.budget_tokens,
                    window_size=req_md.window_size,
                    n_sink_tokens=req_md.n_sink_tokens,
                    evict_current_chunk=req_md.evict_current_chunk,
                    total_prompt_tokens=req_md.total_prompt_tokens,
                ),
            )

            # MAX-reduce across ranks so every worker frees the same block
            # ids: a slot is shared across ranks for one (req, layer, group).
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

            # One numpy scan over every entry, not a Python call per layer.
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

            self._evict_sliding_window_blocks(
                block_table, row_idx, kept_lengths_phys)

    def _postprocess_compress_updates(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Fold pending compression results into the input batch. eff_seq_lens
        becomes the next step's source of truth; the freed block ids go back for
        the scheduler to release.
        """
        for req_id, eff_lens in self.pending_eff_seq_lens.items():
            row = self.input_batch.req_id_to_index.get(req_id)
            if row is None:
                # Removed this step; its KV is already returning to the pool.
                continue
            self.input_batch.effective_seq_lens_cpu[row, :] = eff_lens

        new_eff_seq_lens = dict(self.pending_eff_seq_lens)
        if self.pending_freed_blocks:
            # The dedup matters: no id may reach ``free_blocks_by_ids`` twice.
            freed_block_ids = np.unique(
                np.concatenate(self.pending_freed_blocks)
            )
        else:
            freed_block_ids = np.empty(0, dtype=np.int32)
        # Sliding ids travel on their own channel: they are nulled in place,
        # not freed by a shrink. See ``null_blocks_by_ids``.
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
        """Compression-active context for one step. Yields whether compression
        runs, so the caller knows to invoke the post-forward loop.
        ``_end_compression_step`` fires on exit even on exception.
        """
        active = bool(compression_metadata)
        if active:
            assert self.compressor is not None, (
                "scheduler emitted compression_metadata but the runner has "
                "no KVCompressor; a retention target (compression_ratio or "
                "compression_budget_tokens) must be set."
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
        """Per-row increments for the post-forward ``effective_seq_lens_cpu``
        update, or ``None`` when no row is active.
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

