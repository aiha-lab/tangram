# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

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

# Resolved vLLM model classes validated for ragged paged attention /
# compression. google/gemma-3-12b-it resolves to the multimodal
# Gemma3ForConditionalGeneration (not Gemma3ForCausalLM), so both are listed.
RAGGED_SUPPORTED_ARCHITECTURES = frozenset(
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

    # Ragged paging.
    page_group_size: int | None = 4
    """Head-group size; ``None`` disables ragged paging.
    Must divide ``num_kv_heads``."""
    head_group_cluster_map: str | None = None
    """Head-group cluster map ``.npz`` (``cluster_of`` / ``column_of`` arrays
    built by ``tools/head_group_clustering``) clustering KV heads by retention
    similarity, possibly across layers. Only physical KV placement changes, so
    at no compression the output matches the identity map. Three forms: ``None``
    (default) auto-resolves the bundled map for this model /
    ``compression_scorer`` / ``page_group_size`` /
    ``compression_budget_scope``, falling back to identity when
    none matches; an explicit path loads strictly; ``"identity"`` forces the
    identity (adjacent-head) map. Requires ``page_group_size`` set."""

    # KV cache compression. Generalized over two orthogonal axes: budget
    # scope (``compression_budget_scope``) and score producer
    # (``compression_scorer``).
    compression_ratio: float | None = None
    """Fraction of the KV cache to EVICT (KVpress convention: higher is more
    aggressive), and one of the two compression on/off switches: unset
    (default) or ``0.0`` means no compression; ``0 < r < 1`` enables it
    (requires ``page_group_size``); ``>= 1.0`` is rejected. Mutually exclusive
    with ``compression_budget_tokens`` — a run has ONE retention target. All
    internal math uses the kept fraction, ``compression_keep_ratio``."""
    compression_budget_tokens: int | None = None
    """Fixed KV cache budget in tokens per (layer, head group), and the second
    compression on/off switch (``None`` = off).

    Setting it selects the BUDGET eviction regime instead of the ratio one, the
    formulation the KeyDiff / H2O / SnapKV references use: while the cache fits
    the budget nothing is evicted; the first chunk that would overflow it cuts
    the cache back to the budget by keeping the highest-scoring positions
    outside the sink and the protected recent tail (see
    ``compression_evict_current_chunk``). Unlike ``compression_ratio`` the
    target is absolute, so it holds without knowing the prompt length in
    advance, and no position is locked in — one kept by an earlier chunk
    competes again every chunk, which is what lets a bounded cache admit later,
    more important tokens.

    As with ``compression_ratio``, ``compression_budget_scope`` decides the
    SCOPE the budget is shared over: ``"uniform"`` gives every (layer, head
    group) exactly this length, while ``"layer"`` / ``"global"`` pool it so
    strong groups keep more than weak ones at the same total. The budget applies to
    prompt processing; during decode the cache grows past it again (eviction
    runs once, over the prefill).

    Must exceed ``compression_n_sink_tokens`` plus the protected tail, so there
    is something left to keep."""
    compression_evict_current_chunk: bool = False
    """Budget regime only: whether the chunk just written is itself an eviction
    candidate.

    ``False`` (default) protects the whole fresh chunk, so only positions from
    earlier chunks compete — the reference behaviour, and the more accurate one
    in our measurements. ``True`` protects only the always-kept
    ``compression_window_size`` recent tokens and lets the rest of the fresh
    chunk compete like any other region, which reaches the budget sooner on
    prompts whose chunk size is large relative to the budget. Requires a
    budget: the ratio regime's protected tail is always the recent window."""
    compression_slot_score_source: str = "auto"
    """Budget regime only: where a cached position's score comes from when it
    competes in a later chunk's eviction.

    ``"auto"`` (default, the only setting meant for serving) derives it from the
    scorer, which is the thing that decides it: a scorer whose score is relative
    to the cache (KeyDiff) rescores every live position from the cached keys
    (``"recompute"``), and any other scorer keeps the score each position's own
    chunk produced (``"persist"``). See ``slot_scores.py``.

    Naming a source forces it, which exists for ablations: switching a KeyDiff
    run from a ratio to a budget changes the retention target AND the score
    (chunk-local anchor -> whole-cache anchor) at once, and forcing ``"persist"``
    holds the score at what the ratio regime ranks so the two effects can be
    measured apart. Forcing ``"persist"`` on a rescoring scorer therefore ranks
    positions by scores the method does not specify; forcing ``"recompute"`` on a
    scorer that cannot rescore is rejected. Requires
    ``compression_budget_tokens``: the ratio regime has no cached position to
    score, so a forced source there would silently do nothing."""
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

    # --- Compression: two orthogonal axes (budget scope + scorer) ---
    compression_budget_scope: str = "layer"
    """Axis 1 — the scope the retention budget is balanced over (the rule
    turning eval scores into a per-(layer, group) kept count). One of:

    * ``"layer"`` (default) — each layer gets an equal share of the budget,
      pooled across the head groups within it: strong groups keep more, weak
      ones less. Immune to cross-layer score-scale disparity. Needs a
      within-layer cluster map; TP=1 only.
    * ``"global"`` — one budget pooled across ALL layers and head groups, so
      important groups in any layer can keep more. Sensitive to cross-layer
      score-scale disparity (a layer with systematically larger scores
      monopolises budget). Needs a cross-layer (global) cluster map; TP=1 only.
    * ``"uniform"`` — every (layer, group) keeps the same count
      ``floor(compression_keep_ratio * eval_len)``. Positions still differ per
      head; only the kept *count* is uniform. The only scope available under
      TP>1.

    The accepted set is owned by ``budget_scope.BUDGET_SCOPES``."""
    compression_scorer: str = "snapkv"
    """Axis 2 — score producer, defaulting to ``"snapkv"``. The gate-free
    scorers need no checkpoint: SnapKV scores from observation-window attention
    over the chunk's post-RoPE query/key, KeyDiff from key similarity to the
    chunk's mean key direction, StreamingLLM from token recency (global
    position) alone — a recency baseline, and TOVA from the last query's
    attention averaged across heads (head-uniform). ``"fastkvzip"`` instead uses
    the trained gate over hidden_states (needs a checkpoint)."""
    compression_scorer_options: dict[str, str] = field(default_factory=dict)
    """Axis 2 — settings that only the selected ``compression_scorer``
    understands, as ``{key: value}`` (CLI: ``--compression-scorer-options
    key=value,key=value``, or a JSON object).

    Each scorer DECLARES the settings it accepts, their types, their defaults
    and their help text (``QKScorer.OPTIONS``; see
    ``compression/scorer_options.py``), and this one channel carries them. That
    is deliberate: a scorer-specific configuration field would have to be added
    here, in the CLI, in the ``LLM`` entrypoint and in two factory signatures
    every time a scorer gains a setting, which is precisely what the axis-2
    registry exists to avoid. An unknown key is rejected at startup, naming the
    keys the active scorer does accept.

    Example — reproduce the KeyDiff paper's Eq. (8) anchor instead of the
    unnormalized mean its experiments use::

        --compression-scorer keydiff --compression-scorer-options anchor=normalized

    The scorer's ``OPTIONS`` own the defaults for any key not given here."""

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
            "compression_budget_tokens",
            "compression_evict_current_chunk",
            "compression_slot_score_source",
            "compression_floor_min",
            "compression_chunk_size",
            "compression_window_size",
            "compression_n_sink_tokens",
            "compression_gate_path",
            "compression_budget_scope",
            "compression_scorer",
            "compression_scorer_options",
            "compression_retention_dump",
            # Cluster map relabels physical KV placement only; it does not
            # change the compiled graph shape or kernel selection.
            "head_group_cluster_map",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        # Hash the on/off gate (it changes the forward path), not the
        # continuous ratio, now that ``enable_compression`` is gone.
        factors["compression_enabled"] = self.compression_enabled
        # ``compression_scorer`` itself is excluded above, but its
        # hidden-states-vs-qk axis DOES change the traced graph: the
        # FastKVZip gate wraps every attention block's forward with a
        # ``vllm::tangram_gate_capture`` op node, which qk scorers (SnapKV,
        # ...) do not emit. Without this bit a compile cache written under
        # one scorer kind would be silently reused for the other, dropping
        # (or spuriously adding) the gate-capture nodes.
        factors["gate_capture_installed"] = (
            self.compression_enabled and self.compression_scorer == "fastkvzip"
        )
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

    #: Alias fields into ``compression_scorer_options``, as
    #: ``{scorer: {option name: field name}}``. The scorer's ``OPTIONS`` hold the
    #: real defaults; a test pins the two in agreement so they cannot drift.
    @property
    def resolved_scorer_options(self) -> dict[str, str]:
        """``compression_scorer_options`` with every value normalized to the
        raw string the scorer's option declarations parse."""
        return {key: str(value)
                for key, value in self.compression_scorer_options.items()}

    @property
    def compression_enabled(self) -> bool:
        """Whether KV cache compression runs — either retention target being
        set turns it on: ``compression_ratio > 0`` (unset or ``0.0`` evicts
        nothing, the no-op baseline) or ``compression_budget_tokens`` not None.
        Single source of truth for the gate — consumers read this rather than
        re-deriving the test."""
        return CacheConfig.is_compression_enabled(
            self.compression_ratio, self.compression_budget_tokens)

    @staticmethod
    def is_compression_enabled(ratio: float | None,
                               budget_tokens: int | None) -> bool:
        """The gate, for callers holding the raw values rather than a config."""
        return bool(ratio) or budget_tokens is not None

    @property
    def compression_keep_ratio(self) -> float:
        """Fraction that SURVIVES eviction (``1 - compression_ratio``; ``1.0``
        when unset). The single point converting the public evicted-fraction
        knob into the kept fraction the eviction math is formulated in."""
        return 1.0 - (self.compression_ratio or 0.0)

    @property
    def compression_regime(self) -> str:
        """Which eviction regime the retention target selects (axis 3; see
        ``vllm.v1.attention.compression.eviction_regime``). The two targets are
        mutually exclusive, so the budget's presence decides."""
        return ("budget" if self.compression_budget_tokens is not None
                else "ratio")

    def _derive_head_groups(self) -> None:
        if self.page_group_size is None:
            return
        assert self.num_kv_heads is not None
        assert self.num_hidden_layers is not None
        self.num_head_groups_per_layer = self.num_kv_heads // self.page_group_size
        self.num_head_groups = (
            self.num_head_groups_per_layer * self.num_hidden_layers
        )

    def _validate_budget_target(self) -> None:
        """Reject budgets the eviction geometry could never reach.

        A chunk's kept length is ``sink + selected + protected tail``, so the
        sink and the tail alone must leave room for at least one selected
        position — otherwise the cache would sit above the budget no matter how
        aggressively it evicted, and the "fixed budget" promise would be silently
        false. The tail is the fresh chunk unless the fresh chunk is evictable,
        in which case it is the recent window.
        """
        budget = self.compression_budget_tokens
        if budget is None:
            return
        if budget <= 0:
            raise ValueError(
                f"compression_budget_tokens must be > 0, got {budget}.")
        # The widest tail any chunk can protect, so a budget that admits here
        # admits every prompt length.
        tail = (self.compression_window_size
                if self.compression_evict_current_chunk
                else self.compression_chunk_size)
        floor_needed = self.compression_n_sink_tokens + tail
        if budget <= floor_needed:
            tail_name = ("compression_window_size"
                         if self.compression_evict_current_chunk
                         else "compression_chunk_size")
            raise ValueError(
                f"compression_budget_tokens ({budget}) must exceed "
                f"compression_n_sink_tokens ({self.compression_n_sink_tokens}) "
                f"+ {tail_name} ({tail}) = {floor_needed}: those positions are "
                "kept unconditionally, so a smaller budget is unreachable. "
                "Raise the budget, lower --compression-chunk-size, or set "
                "--compression-evict-current-chunk to protect only the recent "
                "window instead of the whole fresh chunk."
            )

    def _validate_slot_score_source(self) -> None:
        """Reject a forced score provenance that cannot mean what it says.

        The setting is an ablation instrument (see the field docstring), so a
        value that would quietly do nothing is worse than an error: the run
        would look like the ablation and not be it. Two ways that happens — the
        ratio regime, which never scores a cached position at all, and a source
        the active scorer cannot produce.
        """
        from vllm.v1.attention.compression.slot_scores import (
            SLOT_SCORE_SOURCE_AUTO,
            SLOT_SCORE_SOURCE_CHOICES,
        )
        if (self.compression_budget_tokens is None
                and self.compression_evict_current_chunk):
            raise ValueError(
                "compression_evict_current_chunk requires "
                "compression_budget_tokens. The ratio regime's protected tail "
                "is always the recent window, so the setting would have no "
                "effect and the run would look like the ablation without "
                "being it.")
        source = self.compression_slot_score_source
        if source not in SLOT_SCORE_SOURCE_CHOICES:
            raise ValueError(
                f"compression_slot_score_source must be one of "
                f"{SLOT_SCORE_SOURCE_CHOICES}, got {source!r}.")
        if source == SLOT_SCORE_SOURCE_AUTO:
            return
        if self.compression_budget_tokens is None:
            raise ValueError(
                f"compression_slot_score_source={source!r} requires "
                "compression_budget_tokens. Only a budget lets an old position "
                "compete again, so only it needs that position's score; under "
                "the ratio regime a kept position is locked in and the setting "
                "would have no effect. Leave it at 'auto'.")
        if source == "recompute":
            # Checked here too so the failure lands at startup naming both
            # knobs; the factory enforces the same rule against the scorer
            # instance (``rescores_cache``), which stays the authority.
            from vllm.v1.attention.compression.scorer import (
                RESCORING_QK_SCORERS,
            )
            if self.compression_scorer not in RESCORING_QK_SCORERS:
                raise ValueError(
                    f"compression_slot_score_source='recompute' needs a scorer "
                    f"that scores cached positions (one of "
                    f"{RESCORING_QK_SCORERS}), but compression_scorer is "
                    f"{self.compression_scorer!r}. Use 'auto', or pick a "
                    "rescoring scorer.")

    def _validate_extended_fields(self) -> None:
        # Ragged paging.
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
            # Ragged paging is only implemented in the FLASH_ATTN
            if not os.environ.get("VLLM_ATTENTION_BACKEND"):
                os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
                logger.info(
                    "Defaulting VLLM_ATTENTION_BACKEND=FLASH_ATTN (required by "
                    "ragged paging / compression)."
                )

        # Compression. The ratio is an on/off gate, so validate its range
        # unconditionally (an out-of-range value must error, not read as "off").
        # Rejecting >= 1.0 also fails fast for callers still passing the old
        # kept-fraction 1.0 as the no-compression baseline.
        if self.compression_ratio is not None and not (
                0.0 <= self.compression_ratio < 1.0):
            raise ValueError(
                "compression_ratio is the fraction of the KV cache to evict "
                f"and must satisfy 0 <= r < 1, got {self.compression_ratio}. "
                "Leave it unset (or 0.0) to disable compression."
            )
        # The two retention targets are alternative answers to the same
        # question ("how much KV survives"), and they disagree by construction:
        # a ratio scales with the prompt, a budget does not. Reject rather than
        # silently ranking one over the other.
        if (self.compression_budget_tokens is not None
                and self.compression_ratio is not None):
            raise ValueError(
                f"compression_ratio ({self.compression_ratio}) and "
                f"compression_budget_tokens ({self.compression_budget_tokens}) "
                "are mutually exclusive retention targets. Set a ratio to "
                "evict a fraction of the prompt, or a budget to hold the "
                "cache at a fixed token count — and leave compression_ratio "
                "unset when using a budget."
            )
        # Validated outside the compression gate: these knobs name budget-only
        # ablations, and one that is set but silently inert would make a
        # baseline run look like the ablation.
        self._validate_slot_score_source()
        if self.compression_enabled:
            if self.page_group_size is None:
                raise ValueError(
                    "compression requires page_group_size to be set; "
                    "compression operates on top of ragged paging."
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
            self._validate_budget_target()
            # Axis 1 — budget scope. Validated against the registry that
            # ``make_budget_scope`` dispatches on (single source of truth);
            # the local import keeps the torch-backed runtime module out of the
            # config module's import graph.
            from vllm.v1.attention.compression.budget_scope import (
                BUDGET_SCOPES,
            )
            if self.compression_budget_scope not in BUDGET_SCOPES:
                raise ValueError(
                    f"compression_budget_scope must be one of "
                    f"{BUDGET_SCOPES}, got {self.compression_budget_scope!r}."
                )
            # Axis 2 — score producer. Gate-free scorers are owned by the
            # ``scorer`` registry (single source of truth); ``"fastkvzip"`` is
            # the checkpoint-backed hidden_states gate, valid in addition.
            from vllm.v1.attention.compression.scorer import QK_SCORERS
            valid_scorers = ("fastkvzip", *QK_SCORERS)
            if self.compression_scorer not in valid_scorers:
                raise ValueError(
                    f"compression_scorer must be one of {valid_scorers}, "
                    f"got {self.compression_scorer!r}."
                )
            # Axis-2 scorer settings. Resolved (not just parsed) here so an
            # unknown key or an out-of-range value fails at startup with the
            # scorer's accepted keys in the message, rather than deep inside
            # model loading. The scorer class owns the schema; this is only the
            # early check.
            if self.compression_scorer == "fastkvzip":
                if self.compression_scorer_options:
                    raise ValueError(
                        "compression_scorer_options is for the gate-free "
                        "query/key scorers; the fastkvzip gate has no options "
                        "(its behaviour comes from the checkpoint, see "
                        "compression_gate_path).")
            else:
                from vllm.v1.attention.compression.scorer import (
                    get_scorer_options,
                )
                from vllm.v1.attention.compression.scorer_options import (
                    resolve_scorer_options,
                )
                resolve_scorer_options(
                    self.compression_scorer,
                    get_scorer_options(self.compression_scorer),
                    self.resolved_scorer_options)
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

        # Multi-turn rides on top of ragged paging but does not
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

        # Prefix caching cannot represent ragged paging's non-uniform
        # per-(layer, group) block layout or compression's in-place block
        # mutation, so it is disabled. Warn (not info): it is on by default and
        # this affects throughput.
        if (self.compression_enabled or self.page_group_size is not None) and (
            self.enable_prefix_caching
        ):
            feature = (
                "compression" if self.compression_enabled
                else "ragged paging")
            logger.warning(
                "Disabling prefix caching: it is incompatible with %s, which "
                "is enabled. Prefix caching will not be used for this run.",
                feature,
            )
            self.enable_prefix_caching = False

    def verify_model_support(self, architecture: str) -> None:
        """Reject ragged paging / compression on unvalidated models.

        Unsupported architectures fall back to the dense attention path and
        would silently produce wrong outputs, so fail at startup instead.
        ``architecture`` is the resolved vLLM model class.
        """
        if self.page_group_size is None and not self.compression_enabled:
            return
        if architecture in RAGGED_SUPPORTED_ARCHITECTURES:
            return
        feature = (
            "compression" if self.compression_enabled else "ragged paging")
        supported = ", ".join(sorted(RAGGED_SUPPORTED_ARCHITECTURES))
        raise ValueError(
            f"Model architecture '{architecture}' does not support Tangram "
            f"{feature} (ragged paged attention). Supported "
            f"architectures: {supported}. To run this model, disable the "
            f"feature with --page-group-size=None (and leave "
            f"--compression-ratio unset for no compression)."
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

        # The threshold scopes decide budgets at the cluster granularity and
        # would need a cross-rank gather of sharded member scores under TP>1
        # (not implemented) — reject at startup rather than crash at runtime.
        if self.compression_enabled and parallel_config.tensor_parallel_size > 1:
            from vllm.v1.attention.compression.budget_scope import (
                TP1_ONLY_BUDGET_SCOPES,
            )

            if self.compression_budget_scope in TP1_ONLY_BUDGET_SCOPES:
                raise ValueError(
                    f"compression_budget_scope="
                    f"'{self.compression_budget_scope}' supports TP=1 only, "
                    f"got tensor_parallel_size="
                    f"{parallel_config.tensor_parallel_size}. Use "
                    f"compression_budget_scope='uniform' under tensor "
                    f"parallelism.")

    def resolve_head_group_cluster_map(
        self,
        model_config: Any,
        parallel_config: ParallelConfig,
    ) -> None:
        """Freeze ``head_group_cluster_map`` to a concrete path or ``None``
        (identity), once, so the compressor and the attention builder read the
        SAME map and placement can never disagree with scoring. Must run after
        ``verify_with_parallel_config`` (which rejects TP>1 with the threshold
        scopes). Resolves the three field forms (see its docstring) to the two
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
        from vllm.v1.attention.compression.budget_scope import (
            CLUSTER_MAP_SCOPE_BY_BUDGET_SCOPE,
        )

        self.head_group_cluster_map = resolve_bundled_cluster_map(
            scorer=self.compression_scorer,
            model_name=model_config.model,
            page_group_size=self.page_group_size,
            cluster_map_scope=CLUSTER_MAP_SCOPE_BY_BUDGET_SCOPE.get(
                self.compression_budget_scope),
            num_kv_heads=self.num_kv_heads,
            tp_world_size=parallel_config.tensor_parallel_size,
        )
