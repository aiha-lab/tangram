# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
from dataclasses import dataclass, replace as dataclass_replace
from typing import ClassVar

import numpy as np
import torch

from vllm import envs
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
    is_quantized_kv_cache,
)
from vllm.attention.layer import Attention
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.utils.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
from vllm.config import VllmConfig, get_current_vllm_config, get_layers_from_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.ragged_layout import (
    as_virtual_block_view,
    identity_member_maps,
    load_cluster_map,
    member_maps_from_cluster_map,
    member_seq_lens,
    member_virtual_block_table,
    member_virtual_slots,
    physical_member_maps_from_static_cluster_map,
    read_cluster_map_static_layer_ids,
)
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    full_attention_layer_indices,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec, RaggedAttentionSpec

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        from vllm.attention import AttentionType

        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto"]

    @classmethod
    def supports_sink(cls) -> bool:
        # Attention sinks are handled either by FlashAttention 3's native
        # ``s_aux`` (Hopper+) or, on older GPUs, by an exact post-hoc rescale of
        # the output using the returned softmax log-sum-exp
        # (``FlashAttentionImpl._apply_sink_lse_correction``). Both only need the
        # varlen entry point, so sinks are supported wherever it is available —
        # not just on FA3. This lets sink models (e.g. gpt-oss) use the
        # FlashAttention backend on pre-Hopper hardware, which the
        # FlashAttention-only ragged paging path requires.
        return is_flash_attn_varlen_func_available()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # Attention sinks: FlashAttention 3 (Hopper+) handles them natively via
        # the kernel's ``s_aux`` argument. On older GPUs (e.g. A100, where only
        # FA2 is available) the impl recovers the identical result by rescaling
        # the output with the returned log-sum-exp — see
        # ``FlashAttentionImpl._apply_sink_lse_correction``. So sinks are
        # supported on every compute capability FlashAttention itself supports,
        # and no longer gate this backend out (which previously forced
        # sink models such as gpt-oss onto the Triton backend on pre-Hopper
        # hardware, and made ragged paging — FlashAttention-only — unusable
        # for them).
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True

    # Ragged paging fields. Inactive when
    # ``num_head_groups_per_layer == 0``. A sequence is one (req, KV-head)
    # member (member row ``m = L * num_kv_heads + h``), not (req, group).
    #
    # Each attention call builds its own layer's virtual block table on demand
    # (``member_virtual_block_table``); the all-member table is never stored, as
    # it replicates the physical table page_group_size-fold and dominates peak
    # memory at long context. Stored inputs:
    #     cluster_block_table  [num_reqs, num_clusters_total, max_blocks]
    #     clusters_per_layer   [num_layers, num_kv_heads]  (member -> cluster)
    #     cols_per_layer       [num_layers, num_kv_heads]  (member -> column)
    #     layer block table = phys_block[req, cluster] * page_group_size + column
    #
    # ``seq_lens_grouped`` / ``slot_mapping_grouped`` have no block axis, so they
    # keep the all-member form; layout selected by ``ragged_decode_layout``:
    #   False (prefill / mixed) — member-major:
    #     seq_lens_grouped     [num_members_total, num_reqs]
    #     slot_mapping_grouped [num_members_total, num_actual_tokens]
    #     Per-layer slice ``[L*num_kv_heads : (L+1)*num_kv_heads]`` -> varlen.
    #   True (uniform decode, ``max_query_len == 1``) — req-major + layer axis:
    #     seq_lens_grouped     [num_layers, num_reqs, num_kv_heads]
    #     slot_mapping_grouped [num_layers, num_reqs, num_kv_heads]
    num_head_groups_per_layer: int = 0
    cluster_block_table: torch.Tensor | None = None
    clusters_per_layer: torch.Tensor | None = None
    cols_per_layer: torch.Tensor | None = None
    page_group_size: int = 0
    seq_lens_grouped: torch.Tensor | None = None
    slot_mapping_grouped: torch.Tensor | None = None
    # cu_seqlens for the (num_kv_heads × num_reqs) member sequences per
    # layer; shape ``[num_kv_heads × num_reqs + 1]``.
    query_start_loc_grouped: torch.Tensor | None = None
    ragged_decode_layout: bool = False
    # Per-layer ``FlashAttentionMetadata`` overlays for the decode fast
    # path; pre-built so attention avoids ``dataclass_replace`` in the
    # hot loop.
    per_layer_md: list | None = None


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Ragged paging cannot put attention inside a FULL cudagraph:
        # build() allocates fresh metadata tensors every step (virtual block
        # tables, member seq-lens, per-layer overlays), so a captured graph
        # would replay against freed capture-time addresses. Piecewise
        # cudagraphs remain fully supported — the ragged attention op
        # is a splitting op and runs eagerly between captured pieces.
        if isinstance(kv_cache_spec, RaggedAttentionSpec):
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3
        # Ragged paging disables AOT scheduling because each layer
        # sees ``num_kv_heads × num_reqs`` virtual sequences (one per
        # KV-head member), which AOT did not precompute against.
        if isinstance(kv_cache_spec, RaggedAttentionSpec):
            self.aot_schedule = False

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        # Ragged paging adds group-major views to the base metadata;
        # the per-layer slice happens in ``Attention.forward``.
        self._ragged = isinstance(kv_cache_spec, RaggedAttentionSpec)
        if self._ragged:
            assert isinstance(kv_cache_spec, RaggedAttentionSpec)
            self._num_head_groups_per_layer = (
                kv_cache_spec.num_head_groups_per_layer
            )
            # Column-major virtual-block addressing needs the page (column)
            # width and the per-layer KV-head count. See
            # vllm/v1/attention/backends/ragged_layout.py.
            self._page_group_size = kv_cache_spec.page_group_size
            self._num_kv_heads_per_layer = (
                self._num_head_groups_per_layer * self._page_group_size
            )
            # member->(cluster, column) maps, lazily built in ``build()`` once
            # the layer count is known and cached. ``_cluster_map`` is None for
            # the identity map (adjacent-head grouping); a loaded
            # ``(cluster_of, column_of)`` pair makes it a real, possibly
            # cross-layer, cluster map.
            cluster_map_path = self.cache_config.head_group_cluster_map
            self._cluster_map: tuple[torch.Tensor, torch.Tensor] | None = (
                None if cluster_map_path is None
                else load_cluster_map(
                    cluster_map_path, self._page_group_size,
                    self._num_kv_heads_per_layer)
            )
            self._member_to_cluster: torch.Tensor | None = None
            self._member_to_col: torch.Tensor | None = None
        else:
            self._num_head_groups_per_layer = 0
            self._page_group_size = 0
            self._num_kv_heads_per_layer = 0

        if self.use_full_cuda_graph and self.aot_schedule:
            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = envs.VLLM_FLASH_ATTN_MAX_NUM_SPLITS_FOR_CUDA_GRAPH

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def _member_maps(
        self, num_layers_local: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached ``(member_to_cluster, member_to_col)`` for ragged paging.

        member row ``m = layer * num_kv_heads_per_layer + head`` -> the cluster
        that head reads/writes and its column within the cluster's page. The
        identity map (``_cluster_map is None``) reproduces adjacent-head
        grouping; a loaded cluster map assigns arbitrary, possibly
        cross-layer, clusters. Built once: both the layer count and the map are
        fixed for the model. See ragged_layout.py.
        """
        if self._member_to_cluster is None:
            if self._cluster_map is None:
                member_to_cluster, member_to_col = identity_member_maps(
                    num_layers_local, self._num_kv_heads_per_layer,
                    self._page_group_size, device)
            else:
                cluster_of, column_of = self._cluster_map
                if cluster_of.shape[0] != num_layers_local:
                    # Sliding-window hybrid: the map is authored over the
                    # compressible (full-attention) layers only. Expand it to
                    # the physical layer layout — static cross-layer clusters
                    # placed at the same block-table rows the compression
                    # executor reorders, plus identity grouping for the sliding
                    # layers (physical_member_maps_from_static_cluster_map).
                    static_layer_ids = self._ragged_static_layer_ids(
                        num_layers_local)
                    self._assert_static_layer_ids_match_map(
                        static_layer_ids, cluster_of.shape[0])
                    cluster_of, column_of = (
                        physical_member_maps_from_static_cluster_map(
                            cluster_of, column_of, static_layer_ids,
                            num_layers_local, self._page_group_size))
                member_to_cluster, member_to_col = member_maps_from_cluster_map(
                    cluster_of.to(device), column_of.to(device))
            self._member_to_cluster = member_to_cluster
            self._member_to_col = member_to_col
        return self._member_to_cluster, self._member_to_col

    def _ragged_static_layer_ids(self, num_layers_local: int) -> list[int]:
        """Physical indices of the full-attention (compressible) layers in this
        KV cache group.

        Uses the shared ``full_attention_layer_indices`` so this set is the SAME
        one the compression engine compresses (``gpu_model_runner.
        _init_compression``) — the two cannot drift. The cluster map's layer
        axis is physical-major, so each layer's position in ``self.layer_names``
        must equal its physical index; that is asserted here so any reordering
        surfaces loudly instead of silently mis-mapping KV."""
        from vllm.model_executor.models.utils import extract_layer_index

        for pos, name in enumerate(self.layer_names):
            phys = extract_layer_index(name)
            if phys != pos:
                raise RuntimeError(
                    f"ragged KV group layer order mismatch: position "
                    f"{pos} is physical layer {phys}. The cluster map's layer "
                    f"axis assumes position == physical layer index.")
        return [i for i in full_attention_layer_indices(self.vllm_config)
                if i < num_layers_local]

    def _assert_static_layer_ids_match_map(
        self, static_layer_ids: list[int], map_num_static: int,
    ) -> None:
        """Guard that the cluster map matches the running model's
        full-attention layers before expanding it to the physical layout.

        When the map records ``static_layer_ids`` (built by
        ``tools/head_group_clustering/build_cluster_map``), require an exact
        match — this catches a map built for a different model whose
        full-attention layers happen to be at different physical indices but the
        same count. Older maps without the field fall back to a count check."""
        recorded = read_cluster_map_static_layer_ids(
            self.cache_config.head_group_cluster_map)
        if recorded is not None:
            if recorded != static_layer_ids:
                raise ValueError(
                    f"head_group_cluster_map was built for full-attention "
                    f"layers {recorded}, but the running model's "
                    f"full-attention layers are {static_layer_ids}.")
        elif len(static_layer_ids) != map_num_static:
            raise ValueError(
                f"head_group_cluster_map has {map_num_static} layers but the "
                f"model has {len(static_layer_ids)} full-attention layers.")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if self.use_full_cuda_graph and num_actual_tokens <= self.max_cudagraph_size:
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if vllm_is_batch_invariant():
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_kv_lens_cpu = (
                common_attn_metadata.query_start_loc_cpu[1:]
                - common_attn_metadata.query_start_loc_cpu[:-1]
            )
            dcp_context_kv_lens_cpu = seq_lens_cpu - query_kv_lens_cpu

            dcp_context_kv_lens_cpu = get_dcp_local_seq_lens(
                dcp_context_kv_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            dcp_context_kv_lens = dcp_context_kv_lens_cpu.to(self.device)
            max_dcp_context_kv_len = dcp_context_kv_lens.max().item()

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            suffix_kv_lens = (seq_lens_cpu[:num_reqs] - common_prefix_len).to(
                self.device, non_blocking=True
            )
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        # Precompute per-step views consumed by
        # ``layer.py::_ragged_attention_impl`` (the body of the
        # ``vllm::unified_attention_ragged`` custom op); layout selected
        # by ``ragged_decode_layout``.
        ragged_decode_layout = False
        if self._ragged:
            # Terminology in this block: a *head-group* and a *cluster* are the
            # same thing — one shared-retention-budget unit owning a set of
            # physical blocks. ("head-group" is the kv-cache-manager term;
            # "cluster" is the layout/compression term; they are synonyms.) Its
            # per-page columns are *members*, one KV head each, so expanding the
            # cluster axis by ``page_group_size`` yields the member axis.
            if block_table_tensor.ndim != 3:
                raise RuntimeError(
                    "ragged path expects 3D block_table_tensor "
                    "[num_reqs, num_head_groups_total, max_blocks_per_req], "
                    f"got shape {tuple(block_table_tensor.shape)}.")
            num_head_groups_total = block_table_tensor.shape[1]
            num_head_groups_per_layer = self._num_head_groups_per_layer
            if num_head_groups_total % num_head_groups_per_layer != 0:
                raise RuntimeError(
                    f"num_head_groups_total ({num_head_groups_total}) is not "
                    "divisible by num_head_groups_per_layer "
                    f"({num_head_groups_per_layer}).")
            num_layers_local = (
                num_head_groups_total // num_head_groups_per_layer)
            if slot_mapping.ndim != 2:
                raise RuntimeError(
                    "ragged path expects 2D slot_mapping "
                    "[num_head_groups_total, num_actual_tokens], "
                    f"got shape {tuple(slot_mapping.shape)}.")
            effective_seq_lens_cpu = (
                common_attn_metadata.effective_seq_lens_cpu)
            ragged_decode_layout = max_query_len == 1

            num_kv_heads_per_layer = self._num_kv_heads_per_layer

            # Per-CLUSTER sequence lengths in group-major form
            # [num_clusters_total, num_reqs]; the decode overlay is just a
            # reshape of this, so both layouts share one construction. With
            # compression the cache holds effective[cluster] post-compression
            # tokens plus this step's chunk; otherwise the full length.
            if effective_seq_lens_cpu is not None:
                query_start_loc_cpu = (
                    common_attn_metadata.query_start_loc_cpu[: num_reqs + 1])
                num_scheduled_np = (
                    query_start_loc_cpu[1:].cpu().numpy()
                    - query_start_loc_cpu[:-1].cpu().numpy()
                )
                effective_np = np.asarray(
                    effective_seq_lens_cpu, dtype=np.int32)
                seq_lens_cluster_np = (
                    effective_np[:num_reqs].T.astype(np.int32)
                    + num_scheduled_np.astype(np.int32)[None, :]
                )
                seq_lens_cluster = torch.from_numpy(
                    seq_lens_cluster_np).to(seq_lens.device).contiguous()
            else:
                seq_lens_cluster = (
                    seq_lens.unsqueeze(0)
                    .expand(num_head_groups_total, -1)
                    .contiguous()
                )

            # Gather per-cluster physical metadata into per-member (per-KV-head)
            # column-major virtual addressing. member row m = layer *
            # num_kv_heads_per_layer + head; member_to_cluster[m] is the
            # (possibly cross-layer) cluster that head reads/writes, member_to_col
            # its column. The identity map reproduces adjacent-head grouping.
            # Gathering on the FLAT cluster axis before the per-layer
            # reshape is what lets a cluster span layers. See
            # vllm/v1/attention/backends/ragged_layout.py.
            member_to_cluster, member_to_col = self._member_maps(
                num_layers_local, seq_lens.device)
            # Cluster block table trimmed to the blocks the batch occupies, plus
            # the static per-layer (cluster, column) maps; each attention call
            # builds its own virtual block table from these on demand.
            # block_table_tensor: [num_reqs, num_clusters_total, max_blocks].
            max_blocks = cdiv(max_seq_len, self.block_size)
            cluster_block_table = block_table_tensor[:, :, :max_blocks]
            clusters_per_layer = member_to_cluster.view(
                num_layers_local, num_kv_heads_per_layer)
            cols_per_layer = member_to_col.view(
                num_layers_local, num_kv_heads_per_layer)
            # slot_mapping / seq_lens member views carry no block axis, so the
            # all-member form is cheap; keep it.
            # slot_mapping: [num_clusters_total, num_actual_tokens].
            slot_mapping_member = member_virtual_slots(
                slot_mapping, member_to_cluster, member_to_col,
                self._page_group_size, self.block_size, cluster_axis=0)
            # seq_lens_cluster: [num_clusters_total, num_reqs].
            seq_lens_member = member_seq_lens(
                seq_lens_cluster, member_to_cluster, cluster_axis=0)

            if ragged_decode_layout:
                # Uniform decode: every (req, member) varlen sequence has
                # length 1 so num_actual_tokens == num_reqs.
                if num_actual_tokens != num_reqs:
                    raise RuntimeError(
                        "uniform-decode ragged path expects "
                        f"num_actual_tokens ({num_actual_tokens}) == num_reqs "
                        f"({num_reqs}).")
                # member axis splits into (layer, head); the per-layer block
                # table is built in the overlay loop below.
                # [num_members_total, num_reqs] as
                # [num_layers, num_kv_heads, num_reqs]
                # → [num_layers, num_reqs, num_kv_heads].
                slot_mapping_grouped = (
                    slot_mapping_member
                    .view(num_layers_local, num_kv_heads_per_layer, num_reqs)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                seq_lens_grouped = (
                    seq_lens_member
                    .view(num_layers_local, num_kv_heads_per_layer, num_reqs)
                    .permute(0, 2, 1)
                    .contiguous()
                )
                # Every (req, member) sequence has length 1, so cu_seqlens is
                # ``[0, 1, ..., num_kv_heads × num_reqs]``.
                query_start_loc_grouped = torch.arange(
                    num_kv_heads_per_layer * num_reqs + 1,
                    device=query_start_loc.device,
                    dtype=query_start_loc.dtype,
                )
            else:
                # Member-major path (prefill / mixed): the data copy in
                # ``_ragged_attention_impl`` is required because
                # (req, member) sequences are not contiguous in any
                # token-major view when sequence lengths vary.
                slot_mapping_grouped = slot_mapping_member
                seq_lens_grouped = seq_lens_member
                # cu_seqlens for the (num_kv_heads_per_layer × num_reqs)
                # member sequences per layer; the layer slice happens in
                # ``layer.py::_ragged_attention_impl``.
                query_start_head = query_start_loc[:num_reqs]
                offsets = torch.arange(
                    num_kv_heads_per_layer,
                    device=query_start_head.device,
                    dtype=query_start_head.dtype,
                ) * num_actual_tokens
                query_start_grouped_head = (
                    query_start_head.unsqueeze(0) + offsets.unsqueeze(1)
                ).reshape(-1)
                query_start_grouped_tail = torch.tensor(
                    [num_kv_heads_per_layer * num_actual_tokens],
                    device=query_start_head.device,
                    dtype=query_start_head.dtype,
                )
                query_start_loc_grouped = torch.cat(
                    [query_start_grouped_head, query_start_grouped_tail])
        else:
            cluster_block_table = None
            clusters_per_layer = None
            cols_per_layer = None
            seq_lens_grouped = None
            slot_mapping_grouped = None
            query_start_loc_grouped = None

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
            num_head_groups_per_layer=self._num_head_groups_per_layer,
            cluster_block_table=cluster_block_table,
            clusters_per_layer=clusters_per_layer,
            cols_per_layer=cols_per_layer,
            page_group_size=self._page_group_size,
            seq_lens_grouped=seq_lens_grouped,
            slot_mapping_grouped=slot_mapping_grouped,
            query_start_loc_grouped=query_start_loc_grouped,
            ragged_decode_layout=ragged_decode_layout,
        )
        if ragged_decode_layout:
            # Pre-build per-layer overlays so the attention layer picks
            # the right metadata with a single list lookup. One sequence per
            # (request, KV-head member) now that members are per-head.
            num_virtual_seqs = num_reqs * self._num_kv_heads_per_layer
            # Shallow ``copy.copy`` + field overwrite, not ``dataclass_replace``:
            # replace() re-runs __init__ over every field, which is pure-Python
            # overhead paid num_layers times per decode step (on the eager
            # critical path). copy.copy clones __dict__ and we overwrite only the
            # five per-layer fields; the rest are shared (read-only) references.
            per_layer_md = []
            for layer_idx in range(num_layers_local):
                layer_md = copy.copy(attn_metadata)
                layer_md.num_actual_tokens = num_virtual_seqs
                # This layer's virtual block table, built on demand:
                # [num_reqs, num_kv_heads, max_blocks] ->
                # [num_reqs * num_kv_heads, max_blocks].
                layer_md.block_table = member_virtual_block_table(
                    cluster_block_table, clusters_per_layer[layer_idx],
                    cols_per_layer[layer_idx], self._page_group_size,
                    cluster_axis=1).reshape(num_virtual_seqs, -1)
                layer_md.seq_lens = seq_lens_grouped[layer_idx].view(
                    num_virtual_seqs)
                layer_md.slot_mapping = slot_mapping_grouped[
                    layer_idx].reshape(-1)
                layer_md.query_start_loc = query_start_loc_grouped
                per_layer_md.append(layer_md)
            attn_metadata.per_layer_md = per_layer_md
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version()
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = vllm_is_batch_invariant()

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        # Per-head attention sink (a learned extra logit, e.g. gpt-oss). The
        # native kernel path (``s_aux``) exists only in FlashAttention 3 on
        # Hopper+. When unavailable we fall back to an exact post-hoc rescale of
        # the output by the softmax log-sum-exp (``_apply_sink_lse_correction``,
        # mirroring the FastKVzip baseline), which works on FA2 / pre-Hopper and
        # on the ragged paging path. ``False`` for sink-less models keeps
        # the original hot path untouched (zero overhead).
        self.sink_via_lse_correction = False
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )
            self.sink_via_lse_correction = not flash_attn_supports_sinks()

    def supports_quant_query_input(self) -> bool:
        return True

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        if attn_metadata.num_head_groups_per_layer > 0:
            # Ragged column-major page: present the cache as the
            # standard single-KV-head paged layout over virtual blocks
            # (virtual_block = physical_block * page_group_size + column).
            # slot_mapping and block_table are already in virtual coordinates,
            # so reshape_and_cache_flash / flash_attn_varlen_func run
            # unchanged. See vllm/v1/attention/backends/ragged_layout.py.
            kv_cache = as_virtual_block_view(kv_cache)
        key_cache, value_cache = kv_cache.unbind(0)

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            if self.dcp_world_size > 1:
                if self.sink_via_lse_correction:
                    raise NotImplementedError(
                        "Attention-sink log-sum-exp correction (pre-Hopper "
                        "FlashAttention) is not implemented for the "
                        "decode-context-parallel path."
                    )
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                result = flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=self.sliding_window,
                    block_table=block_table,
                    softcap=self.logits_soft_cap,
                    scheduler_metadata=scheduler_metadata,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                    num_splits=attn_metadata.max_num_splits,
                    # Feed the native sink only when FA3 handles it in-kernel.
                    # Otherwise ask for the log-sum-exp and fold the sink in
                    # below (exact, pre-Hopper / head-group safe).
                    s_aux=None if self.sink_via_lse_correction else self.sinks,
                    return_softmax_lse=self.sink_via_lse_correction,
                )
                if self.sink_via_lse_correction:
                    # ``out`` is already written in place; ``result`` is the
                    # ``(out, lse)`` tuple. ``lse`` is the natural-log softmax
                    # denominator per (query-head, token); rescaling by
                    # ``sigmoid(lse - sink)`` adds the sink's contribution.
                    _, lse = result
                    self._apply_sink_lse_correction(
                        output[:num_actual_tokens], lse, attn_metadata,
                        num_actual_tokens,
                    )
                return output

        # Cascade attention (rare case).
        if self.sink_via_lse_correction:
            raise NotImplementedError(
                "Attention-sink log-sum-exp correction (pre-Hopper "
                "FlashAttention) is not implemented for the cascade-attention "
                "path."
            )
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def _apply_sink_lse_correction(
        self,
        output: torch.Tensor,
        lse: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        num_actual_tokens: int,
    ) -> None:
        """Fold a per-head attention sink into ``output`` in place.

        FlashAttention computes attention over the real keys only. A learned
        per-head *sink* is an extra logit added to the softmax denominator, so
        its exact effect is a post-hoc rescale of the output:

            ``o_with_sink = o * sigmoid(lse - sink)``

        where ``lse`` is the natural-log softmax denominator FlashAttention
        returns (``return_softmax_lse=True``). This mirrors the FastKVzip
        baseline (``prefill/attention/attn.py``) and is numerically exact, so it
        replaces the FA3-only native ``s_aux`` path on pre-Hopper GPUs.

        Two output layouts are handled, distinguished by the ragged paging
        metadata. ``lse`` always arrives as ``[query_heads, tokens]`` (the
        FlashAttention varlen convention), so it is transposed to align with the
        token-major ``output`` rows.

        Args:
            output: attention output slice, viewed as
                ``[num_actual_tokens, query_heads_per_seq, head_size]``,
                modified in place.
            lse: ``[query_heads_per_seq, num_actual_tokens]`` log-sum-exp.
            attn_metadata: provides the head-group layout flags.
            num_actual_tokens: number of varlen rows written by the kernel.
        """
        sinks = self.sinks.to(torch.float32)

        if attn_metadata.num_head_groups_per_layer > 0:
            # Head-group (column-major) paging: each varlen sequence is one KV
            # head carrying its GQA group of query heads. ``output`` rows are
            # member-major, so a row encodes (kv_head, token) and a column is
            # the query head within that group; the global query head is
            # ``kv_head * query_heads_per_kv + column``. The row→kv_head map
            # depends on how
            # vllm/attention/layer.py::_ragged_attention_impl
            # flattened the tensors:
            #   - uniform decode fast path: row = token * num_kv_heads + kv_head
            #     (kv_head varies fastest)        -> tile the [kv, head] grid
            #   - prefill / mixed member-major  : row = kv_head * tokens + token
            #     (token varies fastest)          -> repeat each kv row in place
            query_heads = self.num_queries_per_kv
            num_kv_heads = self.num_kv_heads
            tokens_per_member = num_actual_tokens // num_kv_heads
            sink_grid = sinks.view(num_kv_heads, query_heads)
            if attn_metadata.ragged_decode_layout:
                sink_rows = sink_grid.repeat(tokens_per_member, 1)
            else:
                sink_rows = sink_grid.repeat_interleave(tokens_per_member, dim=0)
            sink_aligned = sink_rows
        else:
            # Standard token-major paging: row = token, column = global query
            # head, so the sink broadcasts directly over the token axis.
            query_heads = self.num_heads
            sink_aligned = sinks.view(1, query_heads)

        # ``lse`` -> [tokens, query_heads]; factor matches ``output`` rows.
        factor = torch.sigmoid(lse.transpose(0, 1) - sink_aligned)
        out = output.view(num_actual_tokens, query_heads, self.head_size)
        # Match the baseline's float32 accumulation before casting back.
        out.copy_((out.to(torch.float32) * factor.unsqueeze(-1)).to(out.dtype))

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        # FA returns LSE in shape [ H, B ] but cp_lse_ag_out_rs wants [ B, H ]
        context_attn_out_cor, context_lse_cor = cp_lse_ag_out_rs(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=None,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        # For encoder attention, process FP8 quantization if needed
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if vllm_is_batch_invariant() else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
