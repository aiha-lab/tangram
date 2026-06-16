# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, SkipValidation, field_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import get_cpu_memory

if TYPE_CHECKING:
    from vllm.config.parallel import ParallelConfig
else:
    ParallelConfig = Any

logger = init_logger(__name__)

BlockSize = Literal[1, 8, 16, 32, 64, 128, 256]
CacheDType = Literal[
    "auto",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
]
MambaDType = Literal["auto", "float32"]
PrefixCachingHashAlgo = Literal["sha256", "sha256_cbor"]
KVOffloadingBackend = Literal["native", "lmcache"]


@config
@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: SkipValidation[BlockSize] = None  # type: ignore
    """Size of a contiguous cache block in number of tokens. On CUDA devices,
    only block sizes up to 32 are supported.

    This config has no static default. If left unspecified by the user, it will
    be set in `Platform.check_and_update_config()` based on the current
    platform."""
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    """The fraction of GPU memory to be used for the model executor, which can
    range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
    utilization. If unspecified, will use the default value of 0.9. This is a
    per-instance limit, and only applies to the current vLLM instance. It does
    not matter if you have another vLLM instance running on the same GPU. For
    example, if you have two vLLM instances running on the same GPU, you can
    set the GPU memory utilization to 0.5 for each instance."""
    swap_space: float = Field(default=4, ge=0)
    """Size of the CPU swap space per GPU (in GiB)."""
    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8_inc).
    Some models (namely DeepSeekV3.2) default to fp8, set to bfloat16 to use
    bfloat16 instead, this is an invalid option for models that do not default
    to fp8.
    """
    is_attention_free: bool = False
    """Whether the model is attention-free. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    num_gpu_blocks_override: int | None = None
    """Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks`
    if specified. Does nothing if `None`. Used for testing preemption."""
    sliding_window: int | None = None
    """Sliding window size for the KV cache. This is primarily set in
    `ModelConfig` and that value should be manually duplicated here."""
    enable_prefix_caching: bool = True
    """Whether to enable prefix caching."""
    prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"
    """Set the hash algorithm for prefix caching:\n
    - "sha256" uses Pickle for object serialization before hashing.\n
    - "sha256_cbor" provides a reproducible, cross-language compatible hash. It
    serializes objects using canonical CBOR and hashes them with SHA-256."""
    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    """
    calculate_kv_scales: bool = False
    """This enables dynamic calculation of `k_scale` and `v_scale` when
    kv_cache_dtype is fp8. If `False`, the scales will be loaded from the model
    checkpoint if available. Otherwise, the scales will default to 1.0."""
    cpu_kvcache_space_bytes: int | None = None
    """(CPU backend only) CPU key-value cache space."""
    mamba_page_size_padded: int | None = None
    """ Optional override for mamba page size; used by hybrid mamba/attention
    models to ensure exact alignment with attention page size."""
    mamba_block_size: int | None = Field(default=None, gt=0)
    """Size of a contiguous cache block in number of tokens for mamba cache.
    Can be set only when prefix caching is enabled.
    Value must be a multiple of 8 to align with causal_conv1d kernel."""
    mamba_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (both the conv as well as the
    ssm state). If set to 'auto', the data type will be inferred from the model
    config."""
    mamba_ssm_cache_dtype: MambaDType = "auto"
    """The data type to use for the Mamba cache (ssm state only, conv state will
    still be controlled by mamba_cache_dtype). If set to 'auto', the data type
    for the ssm state will be determined by mamba_cache_dtype."""

    # Will be set after profiling.
    num_gpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for GPU memory."""
    num_cpu_blocks: int | None = field(default=None, init=False)
    """The number of blocks to allocate for CPU memory."""

    kv_sharing_fast_prefill: bool = False
    """This feature is work in progress and no prefill optimization takes place
    with this flag enabled currently.

    In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254),
    some layers can skip tokens corresponding to prefill. This flag enables
    attention metadata for eligible layers to be overridden with metadata
    necessary for implementing this optimization in some models (e.g. Gemma3n)
    """

    kv_cache_memory_bytes: int | None = None
    """Size of KV Cache per GPU in bytes. By default, this is set to None
    and vllm can automatically infer the kv cache size based on
    gpu_memory_utilization. However, users may want to manually specify
    the kv cache memory size. kv_cache_memory_bytes allows more fine-grain
    control of how much memory gets used when compared with using
    gpu_memory_utilization. Note that kv_cache_memory_bytes
    (when not-None) ignores gpu_memory_utilization"""

    kv_offloading_size: float | None = None
    """Size of the KV cache offloading buffer in GiB. When TP > 1, this is
    the total buffer size summed across all TP ranks. By default, this is set
    to None, which means no KV offloading is enabled. When set with
    kv_offloading_backend, vLLM will enable KV cache offloading to CPU"""

    kv_offloading_backend: KVOffloadingBackend | None = None
    """The backend to use for KV cache offloading. Supported backends include
    'native' (vLLM native CPU offloading), 'lmcache' This option must be used
    together with kv_offloading_size."""

    # Head-group paging.
    page_group_size: int | None = 4
    """Head-group size; ``None`` disables head-group paging.
    Must divide ``num_kv_heads``."""
    head_group_cluster_map: str | None = None
    """Path to a head-group cluster map ``.npz`` with integer arrays
    ``cluster_of`` and ``column_of`` (built by ``tools/head_group_clustering``).
    Default ``None`` uses the identity map (adjacent-head grouping). When set,
    KV heads are clustered by retention-budget similarity (possibly across
    layers); requires ``page_group_size`` set and matching the map's
    ``page_group_size``. Only the physical placement of each head's KV changes,
    so at no compression the output is identical to the identity map."""

    # KV cache compression. Generalized over two orthogonal axes: selection
    # level (``compression_level``) and score producer (``compression_scorer``).
    enable_compression: bool = False
    """Enable KV cache compression (head-group paging required)."""
    compression_ratio: float = 0.3
    """Fraction of tokens kept per chunk; each chunk keeps
    ``floor(ratio * re_eval_size)`` tokens."""
    compression_window_size: int = 32
    """Recent tokens always kept during scoring."""
    compression_n_sink_tokens: int = 4
    """Prefix sink tokens always kept during scoring."""
    compression_floor_min: int = 512
    """Per-(layer, group) ``kept_lengths`` floor in tokens; 0 disables it."""
    compression_chunk_size: int = 2048
    """Chunk size for compression-aware chunked prefill (independent from
    ``long_prefill_token_threshold``)."""
    compression_gate_path: str = "fastkvzip"
    """Gate checkpoint (used only when ``compression_scorer == "fastkvzip"``).
    The default ``"fastkvzip"`` sentinel triggers a HuggingFace Hub download
    from ``hmkim97/tangram-gate``; a local path is also accepted."""

    # --- Compression: two orthogonal axes (selection level + scorer) ---
    compression_level: str = "crosslayer_head"
    """Axis 1 — selection level (the rule turning eval scores into a per-(layer,
    group) kept count). Named ``{scope}_{granularity}`` over two orthogonal axes:
    threshold scope (``crosslayer`` global vs ``perlayer``) and calibration
    granularity (``head`` -> max-pool inflates the budget; ``cluster`` -> exact
    budget). One of:

    * ``"crosslayer_head"`` (default) — a single CROSS-layer global threshold,
      head-calibrated (reference ``pair``). Each (layer, group) keeps a divergent
      count: strong clusters keep more, weak ones less. Sensitive to cross-layer
      score-scale disparity (a layer with systematically larger scores
      monopolises budget).
    * ``"perlayer_head"`` — a SEPARATE threshold per layer (AdaKV-style),
      head-calibrated, so every layer keeps its own top ``compression_ratio``
      fraction while heads within a layer still diverge. Immune to cross-layer
      scale disparity; the validated pairing for ``compression_scorer ==
      "expected_attention"`` (its kvpress reference uses AdaKV per-layer budgets).
    * ``"crosslayer_cluster"`` — cross-layer global threshold, cluster-calibrated
      (the exact-budget counterpart of ``"crosslayer_head"``: cross-layer block
      sharing without max-pool inflation). Needs a cross-layer (global) cluster
      map; shares ``"crosslayer_head"``'s scale-disparity sensitivity; TP=1 only.
    * ``"perlayer_cluster"`` — per-layer threshold, cluster-calibrated: the budget
      is decided at the cluster (shared-block) granularity so the physical KV is
      exactly ``compression_ratio`` (no max-pool inflation). Needs a within-layer
      cluster map; TP=1 only.
    * ``"uniform"`` — a uniform count ``floor(compression_ratio * eval_len)`` per
      (layer, group) (reference ``pair-head``). Positions still differ per head;
      only the kept *count* is uniform.

    The accepted set is owned by ``selection_level.SELECTION_LEVELS``."""
    compression_scorer: str = "fastkvzip"
    """Axis 2 — score producer. ``"fastkvzip"`` uses the trained gate over
    hidden_states (needs a checkpoint). ``"snapkv"``, ``"keydiff"``,
    ``"streamingllm"``, and ``"tova"`` are gate-free and need no checkpoint:
    SnapKV scores from observation-window attention over the chunk's post-RoPE
    query/key, KeyDiff from key similarity to the chunk's mean key direction,
    StreamingLLM from token recency (global position) alone — a recency
    baseline, and TOVA from the last query's attention averaged across heads
    (head-uniform)."""
    compression_snap_window: int = 32
    """SnapKV observation window: number of trailing queries used to score a
    chunk. Distinct from ``compression_window_size`` (the always-kept recent
    region). Auto-shrinks to 16 for short chunks (< 1000), matching the
    reference. Only used when ``compression_scorer == "snapkv"``."""
    compression_snap_kernel: int = 7
    """SnapKV max-pool1d smoothing kernel size (odd). Only used when
    ``compression_scorer == "snapkv"``."""
    compression_ea_use_covariance: bool = True
    """ExpectedAttention: add the query covariance term to the expected
    attention logit (kvpress default True). Only used when
    ``compression_scorer == "expected_attention"``."""
    compression_ea_use_vnorm: bool = True
    """ExpectedAttention: reweight the expected attention by the value norm
    (kvpress default True). Only used when
    ``compression_scorer == "expected_attention"``."""
    compression_ea_n_future_positions: int = 512
    """ExpectedAttention: number of future decode positions whose RoPE rotation
    is averaged to anticipate where future queries attend (kvpress default
    512). Only used when ``compression_scorer == "expected_attention"``."""
    compression_ea_epsilon: float = 1e-2
    """ExpectedAttention: constant added before the value-norm reweighting;
    score is ``(prob + epsilon) * ||value||`` (kvpress reference default 1e-2).
    The epsilon floor lets the low-probability tail of keys fall back to
    value-norm ranking instead of being ordered by near-zero softmax noise; on
    RULER this is what keeps NIAH recall from collapsing. It is only beneficial
    paired with a per-layer budget (``compression_level ==
    "perlayer_head"``): under the cross-layer global threshold the
    uneven per-layer value-norm scale that epsilon introduces biases the budget
    toward high-norm layers. Only used when ``compression_scorer ==
    "expected_attention"``."""

    compression_retention_dump: str | None = None
    """Offline profiling only. When set to a directory path, the engine attaches
    a retention observer that writes each per-request keep decision into that
    directory (one ``.npz`` per decision). Used by the head-group clustering
    retention profiler (``tools/head_group_clustering/build_profile.py
    --backend vllm``) to derive a per-(layer, head) retention profile from the
    engine's own keep decision — the only approach that works for eager-only
    models whose query/key/value cannot be captured in HuggingFace transformers.
    Set ``page_group_size=1`` so each (layer, group) is a single head. ``None``
    in production: no observer, no extra work. See
    ``vllm.v1.attention.compression.profiling``."""

    # Multi-turn serving.
    multi_turn: bool = False
    """Enable multi-turn auto-advance. With this on, a request can carry
    ``multi_turn_token_ids`` and the scheduler advances turns automatically."""

    # Engine-derived; do not set manually.
    num_kv_heads: int | None = field(default=None, init=False)
    """Total KV head count; populated by ``derive_from_model``."""
    num_hidden_layers: int | None = field(default=None, init=False)
    """Total transformer layer count; populated by ``derive_from_model``."""
    num_head_groups_per_layer: int | None = field(default=None, init=False)
    """``num_kv_heads // page_group_size``."""
    num_head_groups: int | None = field(default=None, init=False)
    """``num_head_groups_per_layer * num_hidden_layers`` — the block
    table's group dimension."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        ignored_factors = {
            # Runtime/derived knobs that don't affect compiled graph shape
            "gpu_memory_utilization",
            "swap_space",
            "is_attention_free",
            "num_gpu_blocks_override",
            "enable_prefix_caching",
            "prefix_caching_hash_algo",
            # `cpu_offload_gb` does not use `torch.compile` yet.
            "cpu_offload_gb",
            "cpu_kvcache_space_bytes",
            "mamba_page_size_padded",
            # Post-init/derived counters
            "num_gpu_blocks",
            "num_cpu_blocks",
            # WIP feature toggle not impacting compiled graph shape
            "kv_sharing_fast_prefill",
            # Model-meta / runtime policy fields that don't change kernel
            # selection or compiled graph shape.
            "num_kv_heads",
            "num_hidden_layers",
            "num_head_groups",
            "num_head_groups_per_layer",
            "compression_ratio",
            "compression_floor_min",
            "compression_chunk_size",
            # Cluster map relabels physical KV placement only; it does not
            # change the compiled graph shape or kernel selection.
            "head_group_cluster_map",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    @field_validator("cache_dtype", mode="after")
    @classmethod
    def _validate_cache_dtype(cls, cache_dtype: CacheDType) -> CacheDType:
        if cache_dtype.startswith("fp8"):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor."
            )
        return cache_dtype

    def __post_init__(self) -> None:
        # CacheConfig can be constructed standalone, so head-count-dependent
        # checks run only once ``num_kv_heads`` / ``num_hidden_layers`` are
        # populated.
        self._validate_extended_fields()
        if self.num_kv_heads is not None and self.num_hidden_layers is not None:
            self._derive_head_groups()

    def derive_from_model(self, model_config: Any) -> None:
        """Populate KV head / layer counts from ``model_config`` and re-run
        validation."""
        self.num_kv_heads = model_config.get_total_num_kv_heads()
        self.num_hidden_layers = model_config.get_total_num_hidden_layers()
        self._derive_head_groups()
        self._validate_extended_fields()

    def _derive_head_groups(self) -> None:
        if self.page_group_size is None:
            return
        assert self.num_kv_heads is not None
        assert self.num_hidden_layers is not None
        self.num_head_groups_per_layer = self.num_kv_heads // self.page_group_size
        self.num_head_groups = (
            self.num_head_groups_per_layer * self.num_hidden_layers
        )

    def _validate_extended_fields(self) -> None:
        # Head-group paging.
        if self.page_group_size is not None:
            if self.page_group_size <= 0:
                raise ValueError(
                    f"page_group_size must be > 0, got {self.page_group_size}."
                )
            if self.num_kv_heads is not None:
                if self.num_kv_heads % self.page_group_size != 0:
                    raise ValueError(
                        f"num_kv_heads ({self.num_kv_heads}) must be divisible "
                        f"by page_group_size ({self.page_group_size}). Pick a "
                        f"page_group_size that divides num_kv_heads."
                    )
                if self.num_kv_heads // self.page_group_size < 1:
                    raise ValueError(
                        f"num_head_groups_per_layer must be >= 1; got "
                        f"num_kv_heads={self.num_kv_heads}, "
                        f"page_group_size={self.page_group_size}."
                    )
            # Head-grouped paging is only implemented in the FLASH_ATTN
            if not os.environ.get("VLLM_ATTENTION_BACKEND"):
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
                logger.info(
                    "Defaulting VLLM_ATTENTION_BACKEND=FLASH_ATTN (required by "
                    "head-grouped paging / compression)."
                )

        # Compression.
        if self.enable_compression:
            if self.page_group_size is None:
                raise ValueError(
                    "enable_compression=True requires page_group_size to be "
                    "set; compression operates on top of head-group paging."
                )
            if not (0.0 < self.compression_ratio <= 1.0):
                raise ValueError(
                    f"compression_ratio must satisfy 0 < r <= 1, got "
                    f"{self.compression_ratio}."
                )
            if self.compression_window_size <= 0:
                raise ValueError(
                    f"compression_window_size must be > 0, got "
                    f"{self.compression_window_size}."
                )
            if self.compression_n_sink_tokens < 0:
                raise ValueError(
                    f"compression_n_sink_tokens must be >= 0, got "
                    f"{self.compression_n_sink_tokens}."
                )
            if self.compression_floor_min < 0:
                raise ValueError(
                    f"compression_floor_min must be >= 0, got "
                    f"{self.compression_floor_min}."
                )
            if self.compression_chunk_size <= 0:
                raise ValueError(
                    f"compression_chunk_size must be > 0, got "
                    f"{self.compression_chunk_size}."
                )
            if self.compression_chunk_size <= self.compression_window_size:
                raise ValueError(
                    f"compression_chunk_size ({self.compression_chunk_size}) "
                    f"must be greater than compression_window_size "
                    f"({self.compression_window_size})."
                )
            # Axis 1 — selection level. Validated against the registry that
            # ``make_selection_level`` dispatches on (single source of truth);
            # the local import keeps the torch-backed runtime module out of the
            # config module's import graph.
            from vllm.v1.attention.compression.selection_level import (
                SELECTION_LEVELS,
            )
            if self.compression_level not in SELECTION_LEVELS:
                raise ValueError(
                    f"compression_level must be one of {SELECTION_LEVELS}, "
                    f"got {self.compression_level!r}."
                )
            if self.compression_scorer not in (
                    "fastkvzip", "snapkv", "keydiff", "streamingllm", "tova",
                    "expected_attention"):
                raise ValueError(
                    "compression_scorer must be 'fastkvzip', 'snapkv', "
                    "'keydiff', 'streamingllm', 'tova', or "
                    f"'expected_attention', got {self.compression_scorer!r}."
                )
            # The gate checkpoint is only consumed by the fastkvzip scorer;
            # every other (gate-free) scorer ignores the path.
            if self.compression_scorer == "fastkvzip" and (
                not isinstance(self.compression_gate_path, str)
                or not self.compression_gate_path
            ):
                raise ValueError(
                    "compression_gate_path must be a non-empty string "
                    "(either 'fastkvzip' for HF download or a local path)."
                )
            if self.compression_scorer == "snapkv":
                if self.compression_snap_window <= 0:
                    raise ValueError(
                        f"compression_snap_window must be > 0, got "
                        f"{self.compression_snap_window}."
                    )
                if (self.compression_snap_kernel <= 0
                        or self.compression_snap_kernel % 2 == 0):
                    raise ValueError(
                        f"compression_snap_kernel must be a positive odd "
                        f"integer, got {self.compression_snap_kernel}."
                    )
            if self.compression_scorer == "expected_attention":
                if self.compression_ea_n_future_positions <= 0:
                    raise ValueError(
                        "compression_ea_n_future_positions must be > 0, got "
                        f"{self.compression_ea_n_future_positions}."
                    )
                if self.compression_ea_epsilon < 0:
                    raise ValueError(
                        "compression_ea_epsilon must be >= 0, got "
                        f"{self.compression_ea_epsilon}."
                    )

        # Multi-turn rides on top of head-group paging but does not
        # require compression.
        if self.multi_turn:
            if self.page_group_size is None:
                raise ValueError(
                    "multi_turn=True requires page_group_size to be set."
                )
            if self.enable_prefix_caching:
                raise ValueError(
                    "multi_turn=True is incompatible with "
                    "enable_prefix_caching=True (multi-turn carry-over "
                    "replaces prefix caching)."
                )

        # Prefix caching is incompatible with compression and head-group
        # paging; auto-disable.
        if (self.enable_compression or self.page_group_size is not None) and (
            self.enable_prefix_caching
        ):
            logger.info(
                "Disabling enable_prefix_caching (incompatible with %s).",
                "compression" if self.enable_compression else "head-group paging",
            )
            self.enable_prefix_caching = False

    def verify_with_parallel_config(
        self,
        parallel_config: ParallelConfig,
    ) -> None:
        swap_space_bytes = self.swap_space * GiB_bytes
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = swap_space_bytes * num_gpus_per_node

        msg = (
            f"{cpu_memory_usage / GiB_bytes:.2f} GiB out of the "
            f"{total_cpu_memory / GiB_bytes:.2f} GiB total CPU memory "
            "is allocated for the swap space."
        )
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. %s", msg)
