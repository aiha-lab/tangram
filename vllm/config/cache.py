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

# Resolved vLLM model classes validated for head-grouped paged attention /
# compression. google/gemma-3-12b-it resolves to the multimodal
# Gemma3ForConditionalGeneration (not Gemma3ForCausalLM), so both are listed.
HEAD_GROUPED_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Gemma3ForCausalLM",
        "Gemma3ForConditionalGeneration",
        "GptOssForCausalLM",
    }
)


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
    """Head-group cluster map ``.npz`` (``cluster_of`` / ``column_of`` arrays
    built by ``tools/head_group_clustering``) clustering KV heads by retention
    similarity, possibly across layers. Only physical KV placement changes, so
    at no compression the output matches the identity map. Three forms: ``None``
    (default) auto-resolves the bundled map for this model / ``compression_scorer``
    / ``page_group_size`` / ``compression_level``, falling back to identity when
    none matches; an explicit path loads strictly; ``"identity"`` forces the
    identity (adjacent-head) map. Requires ``page_group_size`` set."""

    # KV cache compression. Generalized over two orthogonal axes: selection
    # level (``compression_level``) and score producer (``compression_scorer``).
    compression_ratio: float = 1.0
    """Fraction of tokens kept per chunk (``floor(ratio * re_eval_size)``), and
    the compression on/off switch: ``1.0`` (default) keeps everything (no
    compression), ``< 1.0`` enables compression (requires ``page_group_size``).
    Must satisfy ``0 < compression_ratio <= 1``; see ``compression_enabled``."""
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
    compression_level: str = "crosslayer_cluster"
    """Axis 1 — selection level (the rule turning eval scores into a per-(layer,
    group) kept count). Named ``{scope}_{granularity}`` over two orthogonal axes:
    threshold scope (``crosslayer`` global vs ``perlayer``) and calibration
    granularity (``head`` -> max-pool inflates the budget; ``cluster`` -> exact
    budget). The cluster-calibrated levels are TP=1 only; under TP>1 they are
    downgraded to their head-calibrated counterpart of the same scope (see
    ``verify_with_parallel_config``). One of:

    * ``"crosslayer_head"`` — a single CROSS-layer global threshold,
      head-calibrated (reference ``pair``). Each (layer, group) keeps a divergent
      count: strong clusters keep more, weak ones less. Sensitive to cross-layer
      score-scale disparity (a layer with systematically larger scores
      monopolises budget). Tangram's historical default.
    * ``"perlayer_head"`` — a SEPARATE threshold per layer (AdaKV-style),
      head-calibrated, so every layer keeps its own top ``compression_ratio``
      fraction while heads within a layer still diverge. Immune to cross-layer
      scale disparity; the validated pairing for ``compression_scorer ==
      "expected_attention"`` (its kvpress reference uses AdaKV per-layer budgets).
    * ``"crosslayer_cluster"`` (default) — cross-layer global threshold,
      cluster-calibrated (the exact-budget counterpart of ``"crosslayer_head"``:
      cross-layer block sharing without max-pool inflation). Needs a cross-layer
      (global) cluster map; shares ``"crosslayer_head"``'s scale-disparity
      sensitivity; TP=1 only.
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
            # Runtime KV-eviction policy (scheduler / compressor, outside the
            # compiled forward) — no effect on graph shape. page_group_size
            # stays hashed (it changes graph / backend); the compression on/off
            # bit is hashed separately below via ``compression_enabled``.
            "compression_ratio",
            "compression_floor_min",
            "compression_chunk_size",
            "compression_window_size",
            "compression_n_sink_tokens",
            "compression_gate_path",
            "compression_level",
            "compression_scorer",
            "compression_snap_window",
            "compression_snap_kernel",
            "compression_ea_use_covariance",
            "compression_ea_use_vnorm",
            "compression_ea_n_future_positions",
            "compression_ea_epsilon",
            "compression_retention_dump",
            # Cluster map relabels physical KV placement only; it does not
            # change the compiled graph shape or kernel selection.
            "head_group_cluster_map",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        # Hash the on/off gate (it forces eager / changes the forward path), not
        # the continuous ratio, now that ``enable_compression`` is gone.
        factors["compression_enabled"] = self.compression_enabled
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

    @property
    def compression_enabled(self) -> bool:
        """Whether KV cache compression runs: ``compression_ratio < 1.0``
        (``1.0`` keeps every token, the no-op baseline). Single source of truth
        for the gate — consumers read this rather than re-deriving the test."""
        return self.compression_ratio < 1.0

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

        # Compression. The ratio is the on/off gate, so validate its range
        # unconditionally (an out-of-range value must error, not read as "off").
        if not (0.0 < self.compression_ratio <= 1.0):
            raise ValueError(
                f"compression_ratio must satisfy 0 < r <= 1, got "
                f"{self.compression_ratio}."
            )
        if self.compression_enabled:
            if self.page_group_size is None:
                raise ValueError(
                    "compression (compression_ratio < 1.0) requires "
                    "page_group_size to be set; compression operates on top of "
                    "head-group paging."
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

        # Prefix caching cannot represent head-group paging's non-uniform
        # per-(layer, group) block layout or compression's in-place block
        # mutation, so it is disabled. Warn (not info): it is on by default and
        # this affects throughput.
        if (self.compression_enabled or self.page_group_size is not None) and (
            self.enable_prefix_caching
        ):
            feature = (
                "compression" if self.compression_enabled
                else "head-group paging")
            logger.warning(
                "Disabling prefix caching: it is incompatible with %s, which "
                "is enabled. Prefix caching will not be used for this run.",
                feature,
            )
            self.enable_prefix_caching = False

    def verify_model_support(self, architecture: str) -> None:
        """Reject head-grouped paging / compression on unvalidated models.

        Unsupported architectures fall back to the dense attention path and
        would silently produce wrong outputs, so fail at startup instead.
        ``architecture`` is the resolved vLLM model class.
        """
        if self.page_group_size is None and not self.compression_enabled:
            return
        if architecture in HEAD_GROUPED_SUPPORTED_ARCHITECTURES:
            return
        feature = (
            "compression" if self.compression_enabled else "head-group paging")
        supported = ", ".join(sorted(HEAD_GROUPED_SUPPORTED_ARCHITECTURES))
        raise ValueError(
            f"Model architecture '{architecture}' does not support Tangram "
            f"{feature} (head-grouped paged attention). Supported "
            f"architectures: {supported}. To run this model, disable the "
            f"feature with --page-group-size=None (and --compression-ratio=1.0 "
            f"for no compression)."
        )

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

        # Cluster-calibrated levels need a cross-rank member-score gather not yet
        # implemented, so under TP>1 downgrade to the same-scope head-calibrated
        # level (which all-gathers and works under TP) rather than reject.
        # TODO: implement the gather and run cluster levels under TP directly.
        if self.compression_enabled and parallel_config.tensor_parallel_size > 1:
            from vllm.v1.attention.compression.selection_level import (
                TP1_ONLY_SELECTION_LEVELS,
                TP_FALLBACK_LEVEL,
            )

            # A TP1-only level with no registered fallback is a startup error,
            # not a late runtime crash (guards drift between the two tables).
            if self.compression_level in TP1_ONLY_SELECTION_LEVELS:
                fallback = TP_FALLBACK_LEVEL.get(self.compression_level)
                if fallback is None:
                    raise ValueError(
                        f"compression_level='{self.compression_level}' is TP=1 "
                        f"only but has no TP>1 fallback registered in "
                        f"TP_FALLBACK_LEVEL; add one or select a head-calibrated "
                        f"level for tensor_parallel_size="
                        f"{parallel_config.tensor_parallel_size}.")
                logger.warning(
                    "compression_level='%s' is TP=1 only (got "
                    "tensor_parallel_size=%d); downgrading to '%s' (same "
                    "threshold scope, head-calibrated) for this run.",
                    self.compression_level,
                    parallel_config.tensor_parallel_size, fallback)
                self.compression_level = fallback

    def resolve_head_group_cluster_map(
        self,
        model_config: Any,
        parallel_config: ParallelConfig,
    ) -> None:
        """Freeze ``head_group_cluster_map`` to a concrete path or ``None``
        (identity), once, so the compressor and the attention builder read the
        SAME map and placement can never disagree with scoring. Must run after
        ``verify_with_parallel_config`` (which may downgrade ``compression_level``
        under TP). Resolves the three field forms (see its docstring) to the two
        the consumers understand: a path or ``None``.
        """
        raw = self.head_group_cluster_map
        if raw == "identity":  # explicit opt-out -> identity map
            self.head_group_cluster_map = None
            return
        if raw is not None:  # explicit path: load_cluster_map loads it strictly
            return
        # raw is None -> auto-resolve. Skip when the map is never exercised.
        if not self.compression_enabled or self.page_group_size is None:
            return

        from vllm.v1.attention.backends.cluster_map_resolver import (
            resolve_bundled_cluster_map,
        )
        from vllm.v1.attention.compression.selection_level import (
            CLUSTER_MAP_SCOPE_BY_LEVEL,
        )

        self.head_group_cluster_map = resolve_bundled_cluster_map(
            scorer=self.compression_scorer,
            model_name=model_config.model,
            page_group_size=self.page_group_size,
            cluster_map_scope=CLUSTER_MAP_SCOPE_BY_LEVEL.get(
                self.compression_level),
            num_kv_heads=self.num_kv_heads,
            tp_world_size=parallel_config.tensor_parallel_size,
        )
