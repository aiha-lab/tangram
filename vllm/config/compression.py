# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup validation for the ragged-paging and KV-compression settings.

Every function here reads a ``CacheConfig`` and raises ``ValueError`` on a
combination the engine cannot honour. The settings themselves stay declared on
``CacheConfig``: the CLI derives each ``--compression-*`` option's help text
from the docstring under its field in that class's own source, so the fields
cannot leave it. Only the rules about them live here.

The rules exist because most of these settings fail silently rather than
loudly. A budget smaller than the positions kept unconditionally would hold the
cache above its "fixed budget" forever; a forced score provenance the active
scorer cannot produce would run an ablation that is not the one named; a model
outside the validated set falls back to dense attention and returns wrong
output. Each is a wrong number rather than a crash, so it is rejected at
startup instead.

The runtime registries are the authority on which values exist (budget scopes,
scorers, score sources) and are imported inside the functions: this is a config
module, and must not pull the torch-backed compression package into the config
import graph.
"""
import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.cache import CacheConfig

logger = init_logger(__name__)

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


def validate_budget_target(cfg: "CacheConfig") -> None:
    """Reject budgets the eviction geometry could never reach.

    A chunk's kept length is ``sink + selected + protected tail``, so the
    sink and the tail alone must leave room for at least one selected
    position — otherwise the cache would sit above the budget no matter how
    aggressively it evicted, and the "fixed budget" promise would be silently
    false. The tail is the fresh chunk unless the fresh chunk is evictable,
    in which case it is the recent window.
    """
    budget = cfg.compression_budget_tokens
    if budget is None:
        return
    if budget <= 0:
        raise ValueError(
            f"compression_budget_tokens must be > 0, got {budget}.")
    # The widest tail any chunk can protect, so a budget that admits here
    # admits every prompt length.
    tail = (cfg.compression_window_size
            if cfg.compression_evict_current_chunk
            else cfg.compression_chunk_size)
    floor_needed = cfg.compression_n_sink_tokens + tail
    if budget <= floor_needed:
        tail_name = ("compression_window_size"
                     if cfg.compression_evict_current_chunk
                     else "compression_chunk_size")
        raise ValueError(
            f"compression_budget_tokens ({budget}) must exceed "
            f"compression_n_sink_tokens ({cfg.compression_n_sink_tokens}) "
            f"+ {tail_name} ({tail}) = {floor_needed}: those positions are "
            "kept unconditionally, so a smaller budget is unreachable. "
            "Raise the budget, lower --compression-chunk-size, or set "
            "--compression-evict-current-chunk to protect only the recent "
            "window instead of the whole fresh chunk."
        )


def validate_slot_score_source(cfg: "CacheConfig") -> None:
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
    if (cfg.compression_budget_tokens is None
            and cfg.compression_evict_current_chunk):
        raise ValueError(
            "compression_evict_current_chunk requires "
            "compression_budget_tokens. The ratio regime's protected tail "
            "is always the recent window, so the setting would have no "
            "effect and the run would look like the ablation without "
            "being it.")
    source = cfg.compression_slot_score_source
    if source not in SLOT_SCORE_SOURCE_CHOICES:
        raise ValueError(
            f"compression_slot_score_source must be one of "
            f"{SLOT_SCORE_SOURCE_CHOICES}, got {source!r}.")
    if source == SLOT_SCORE_SOURCE_AUTO:
        return
    if cfg.compression_budget_tokens is None:
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
        if cfg.compression_scorer not in RESCORING_QK_SCORERS:
            raise ValueError(
                f"compression_slot_score_source='recompute' needs a scorer "
                f"that scores cached positions (one of "
                f"{RESCORING_QK_SCORERS}), but compression_scorer is "
                f"{cfg.compression_scorer!r}. Use 'auto', or pick a "
                "rescoring scorer.")


def validate_extended_fields(cfg: "CacheConfig") -> None:
    # Ragged paging.
    if cfg.page_group_size is not None:
        if cfg.page_group_size <= 0:
            raise ValueError(
                f"page_group_size must be > 0, got {cfg.page_group_size}."
            )
        if cfg.num_kv_heads is not None:
            if cfg.num_kv_heads % cfg.page_group_size != 0:
                raise ValueError(
                    f"num_kv_heads ({cfg.num_kv_heads}) must be divisible "
                    f"by page_group_size ({cfg.page_group_size}). Pick a "
                    f"page_group_size that divides num_kv_heads."
                )
            if cfg.num_kv_heads // cfg.page_group_size < 1:
                raise ValueError(
                    f"num_head_groups_per_layer must be >= 1; got "
                    f"num_kv_heads={cfg.num_kv_heads}, "
                    f"page_group_size={cfg.page_group_size}."
                )

    # Compression. The ratio is an on/off gate, so validate its range
    # unconditionally (an out-of-range value must error, not read as "off").
    # Rejecting >= 1.0 also fails fast for callers still passing the old
    # kept-fraction 1.0 as the no-compression baseline.
    if cfg.compression_ratio is not None and not (
            0.0 <= cfg.compression_ratio < 1.0):
        raise ValueError(
            "compression_ratio is the fraction of the KV cache to evict "
            f"and must satisfy 0 <= r < 1, got {cfg.compression_ratio}. "
            "Leave it unset (or 0.0) to disable compression."
        )
    # The two retention targets are alternative answers to the same
    # question ("how much KV survives"), and they disagree by construction:
    # a ratio scales with the prompt, a budget does not. Reject rather than
    # silently ranking one over the other.
    if (cfg.compression_budget_tokens is not None
            and cfg.compression_ratio is not None):
        raise ValueError(
            f"compression_ratio ({cfg.compression_ratio}) and "
            f"compression_budget_tokens ({cfg.compression_budget_tokens}) "
            "are mutually exclusive retention targets. Set a ratio to "
            "evict a fraction of the prompt, or a budget to hold the "
            "cache at a fixed token count — and leave compression_ratio "
            "unset when using a budget."
        )
    # Validated outside the compression gate: these knobs name budget-only
    # ablations, and one that is set but silently inert would make a
    # baseline run look like the ablation.
    validate_slot_score_source(cfg)
    if cfg.compression_enabled:
        if cfg.page_group_size is None:
            raise ValueError(
                "compression requires page_group_size to be set; "
                "compression operates on top of ragged paging."
            )
        if cfg.compression_window_size <= 0:
            raise ValueError(
                f"compression_window_size must be > 0, got "
                f"{cfg.compression_window_size}."
            )
        if cfg.compression_n_sink_tokens < 0:
            raise ValueError(
                f"compression_n_sink_tokens must be >= 0, got "
                f"{cfg.compression_n_sink_tokens}."
            )
        if cfg.compression_floor_min < 0:
            raise ValueError(
                f"compression_floor_min must be >= 0, got "
                f"{cfg.compression_floor_min}."
            )
        if cfg.compression_chunk_size <= 0:
            raise ValueError(
                f"compression_chunk_size must be > 0, got "
                f"{cfg.compression_chunk_size}."
            )
        if cfg.compression_chunk_size <= cfg.compression_window_size:
            raise ValueError(
                f"compression_chunk_size ({cfg.compression_chunk_size}) "
                f"must be greater than compression_window_size "
                f"({cfg.compression_window_size})."
            )
        validate_budget_target(cfg)
        # Axis 1 — budget scope. Validated against the registry that
        # ``make_budget_scope`` dispatches on (single source of truth);
        # the local import keeps the torch-backed runtime module out of the
        # config module's import graph.
        from vllm.v1.attention.compression.budget_scope import (
            BUDGET_SCOPES,
        )
        if cfg.compression_budget_scope not in BUDGET_SCOPES:
            raise ValueError(
                f"compression_budget_scope must be one of "
                f"{BUDGET_SCOPES}, got {cfg.compression_budget_scope!r}."
            )
        # Axis 2 — score producer. Gate-free scorers are owned by the
        # ``scorer`` registry (single source of truth); ``"fastkvzip"`` is
        # the checkpoint-backed hidden_states gate, valid in addition.
        from vllm.v1.attention.compression.scorer import QK_SCORERS
        valid_scorers = ("fastkvzip", *QK_SCORERS)
        if cfg.compression_scorer not in valid_scorers:
            raise ValueError(
                f"compression_scorer must be one of {valid_scorers}, "
                f"got {cfg.compression_scorer!r}."
            )
        # Axis-2 scorer settings. Resolved (not just parsed) here so an
        # unknown key or an out-of-range value fails at startup with the
        # scorer's accepted keys in the message, rather than deep inside
        # model loading. The scorer class owns the schema; this is only the
        # early check.
        if cfg.compression_scorer == "fastkvzip":
            if cfg.compression_scorer_options:
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
                cfg.compression_scorer,
                get_scorer_options(cfg.compression_scorer),
                cfg.resolved_scorer_options)
        # The gate checkpoint is only consumed by the fastkvzip scorer;
        # every other (gate-free) scorer ignores the path.
        if cfg.compression_scorer == "fastkvzip" and (
            not isinstance(cfg.compression_gate_path, str)
            or not cfg.compression_gate_path
        ):
            raise ValueError(
                "compression_gate_path must be a non-empty string "
                "(either 'fastkvzip' for HF download or a local path)."
            )

    # Multi-turn rides on top of ragged paging but does not
    # require compression.
    if cfg.multi_turn:
        if cfg.page_group_size is None:
            raise ValueError(
                "multi_turn=True requires page_group_size to be set."
            )
        if cfg.enable_prefix_caching:
            raise ValueError(
                "multi_turn=True is incompatible with "
                "enable_prefix_caching=True (multi-turn carry-over "
                "replaces prefix caching)."
            )

    # Prefix caching cannot represent ragged paging's non-uniform
    # per-(layer, group) block layout or compression's in-place block
    # mutation, so it is disabled. Warn (not info): it is on by default and
    # this affects throughput.
    if (cfg.compression_enabled or cfg.page_group_size is not None) and (
        cfg.enable_prefix_caching
    ):
        feature = (
            "compression" if cfg.compression_enabled
            else "ragged paging")
        logger.warning(
            "Disabling prefix caching: it is incompatible with %s, which "
            "is enabled. Prefix caching will not be used for this run.",
            feature,
        )
        cfg.enable_prefix_caching = False


def validate_model_support(
    cfg: "CacheConfig", architecture: str
) -> None:
    """Reject ragged paging / compression on unvalidated models.

    Unsupported architectures fall back to the dense attention path and
    would silently produce wrong outputs, so fail at startup instead.
    ``architecture`` is the resolved vLLM model class, which is why
    ``RAGGED_SUPPORTED_ARCHITECTURES`` lists google/gemma-3-12b-it under both
    Gemma3ForConditionalGeneration (what it actually resolves to, being
    multimodal) and Gemma3ForCausalLM.
    """
    if cfg.page_group_size is None and not cfg.compression_enabled:
        return
    if architecture in RAGGED_SUPPORTED_ARCHITECTURES:
        return
    feature = (
        "compression" if cfg.compression_enabled else "ragged paging")
    supported = ", ".join(sorted(RAGGED_SUPPORTED_ARCHITECTURES))
    raise ValueError(
        f"Model architecture '{architecture}' does not support Tangram "
        f"{feature} (ragged paged attention). Supported "
        f"architectures: {supported}. To run this model, disable the "
        f"feature with --page-group-size=None (and leave "
        f"--compression-ratio unset for no compression)."
    )


def default_attention_backend(cfg: "CacheConfig") -> None:
    """Point the attention backend at FlashAttention, the only one that
    implements ragged paging, unless the caller already named one.

    This writes ``VLLM_ATTENTION_BACKEND``, which is process-global and is read
    by every later backend selection, so it belongs to assembling one engine's
    configuration -- not to constructing a ``CacheConfig``, which happens more
    than once per process and at a point where nothing has decided what the run
    uses. An explicit setting is left alone: the default exists for the case
    where the user made no choice.
    """
    if cfg.page_group_size is None:
        return
    if os.environ.get("VLLM_ATTENTION_BACKEND"):
        return

    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    logger.info(
        "Defaulting VLLM_ATTENTION_BACKEND=FLASH_ATTN (required by "
        "ragged paging / compression)."
    )
