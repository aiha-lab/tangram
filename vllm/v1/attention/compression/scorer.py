# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate-free query/key scorer factory (compression axis 2).

The single dispatch point mapping a ``compression_scorer`` name to its
gate-free scorer module (SnapKV, KeyDiff, …). Mirrors ``make_budget_scope``
for axis 1: adding a scorer = one new module + one entry here, with no scorer
branching leaking elsewhere (the runner and compressor stay scorer-agnostic).

FastKVZip is deliberately NOT here — it is checkpoint-backed and consumes
hidden_states, loaded by ``KVCompressor.load_gate_checkpoint``. This factory
covers only the gate-free scorers that consume post-RoPE query/key.
"""
from __future__ import annotations

from typing import Mapping

from torch import nn

from vllm.logger import init_logger
from vllm.v1.attention.compression.keydiff import KeyDiffScorer
from vllm.v1.attention.compression.qk_scorer_base import QKScorer
from vllm.v1.attention.compression.snapkv import SnapKVScorer
from vllm.v1.attention.compression.streamingllm import StreamingLLMScorer
from vllm.v1.attention.compression.expected_attention import (
    ExpectedAttentionScorer,
)
from vllm.v1.attention.compression.scorer_options import (
    ScorerOption,
    describe_scorer_options,
    resolve_scorer_options,
)
from vllm.v1.attention.compression.tova import TOVAScorer

logger = init_logger(__name__)

#: Axis-2 registry: ``compression_scorer`` value -> gate-free scorer class,
#: keyed off each class's ``name`` so the accepted set has one source of truth.
#: Config validation imports ``QK_SCORERS`` and ``build_qk_scorer`` constructs
#: from it, rather than either re-listing the names. Mirrors
#: ``budget_scope._SCOPES`` for axis 1. FastKVZip is intentionally absent —
#: it is the checkpoint-backed hidden_states gate, selected on a separate path.
_QK_SCORERS: dict[str, type[QKScorer]] = {
    cls.name: cls
    for cls in (
        SnapKVScorer,
        KeyDiffScorer,
        StreamingLLMScorer,
        TOVAScorer,
        ExpectedAttentionScorer,
    )
}

#: Valid gate-free ``compression_scorer`` values. Config validation adds the
#: checkpoint-backed ``"fastkvzip"`` to this set (see ``CacheConfig``).
QK_SCORERS: tuple[str, ...] = tuple(_QK_SCORERS)

#: The subset whose score is relative to the cache rather than to the chunk that
#: wrote a position, i.e. those that can score positions already cached (they set
#: ``rescores_cache`` and implement ``score_cached``). Read off the classes
#: so the property is stated once, on the scorer; config validation uses it to
#: reject a forced ``compression_slot_score_source='recompute'`` at startup
#: without constructing a scorer. The checkpoint-backed FastKVZip gate is not a
#: member (its score comes from hidden_states, which the cache does not hold).
RESCORING_QK_SCORERS: tuple[str, ...] = tuple(
    name for name, cls in _QK_SCORERS.items() if cls.rescores_cache)


def _scorer_class(name: str) -> type[QKScorer]:
    scorer_cls = _QK_SCORERS.get(name)
    if scorer_cls is None:
        raise ValueError(
            f"unknown gate-free qk scorer {name!r}; "
            f"expected one of {QK_SCORERS}.")
    return scorer_cls


def get_scorer_options(name: str) -> tuple[ScorerOption, ...]:
    """The settings ``name`` declares, for validation and help text.

    Exposed so configuration can reject a bad option at startup without
    constructing a scorer (which needs model dimensions it does not have).
    """
    return _scorer_class(name).OPTIONS


def build_qk_scorer(
    name: str,
    *,
    num_kv_heads: int,
    num_q_per_kv: int,
    head_size: int,
    options: Mapping[str, str] | None = None,
) -> nn.Module:
    """Construct the gate-free query/key scorer selected by ``name``.

    Every scorer is built through ONE shared contract — ``num_kv_heads`` /
    ``head_size`` / ``num_q_per_kv`` (the per-rank GQA ratio), which a scorer
    that does not need one simply ignores — plus the settings it declared in
    ``OPTIONS``, resolved from ``options`` (see scorer_options.py). There is
    deliberately no per-scorer branch here: adding a scorer, or a setting on
    one, must not require editing this factory, the configuration, the CLI or
    the entrypoint.

    Args:
        name: ``compression_scorer`` value; a registry key.
        options: raw ``{key: value}`` strings for this scorer's declared
            settings. Unknown keys raise rather than being ignored.

    Returns:
        The scorer module, exposing ``consumes`` / ``name`` for the delivery
        dispatch in ``attach_scorers``.
    """
    scorer_cls = _scorer_class(name)
    resolved = resolve_scorer_options(name, scorer_cls.OPTIONS, options)
    logger.info("Compression %s",
                describe_scorer_options(name, scorer_cls.OPTIONS, resolved))
    return scorer_cls(
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_q_per_kv=num_q_per_kv,
        **resolved,
    )
