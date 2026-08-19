# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base contract for gate-free query/key compression scorers (axis 2).

A ``QKScorer`` is a stateless ``nn.Module`` shared across all compressible
layers. It turns one request-chunk's post-RoPE query/key (and optionally value)
into per-KV-head importance scores ``[num_kv_heads, T]`` that the compressor
uses to pick which tokens to keep. Declaring the contract here — rather than in
prose repeated across each scorer — lets the axis-2 registry key scorers off a
single ``name`` and lets the delivery dispatch in ``KVCompressor.attach_scorers``
rely on ``name`` / ``consumes`` being present.

FastKVZip is deliberately not a ``QKScorer``: it is checkpoint-backed and
consumes ``hidden_states`` (not post-RoPE q/k), so it is loaded and delivered on
a separate path. Only the gate-free scorers (SnapKV, KeyDiff, StreamingLLM,
TOVA, ExpectedAttention) implement this base.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from vllm.v1.attention.compression.scorer_options import ScorerOption


class QKScorer(nn.Module, ABC):
    """Gate-free query/key importance scorer (compression axis 2).

    A subclass sets ``name`` (the ``compression_scorer`` value that selects it)
    and implements ``forward``. ``consumes`` records which forward tensors the
    scorer reads; ``"qk"`` — the inner ``Attention``'s post-RoPE query/key — is
    the only value the gate-free scorers use (FastKVZip's ``hidden_states`` path
    is separate), so it is the default and subclasses need not repeat it.
    """

    #: ``compression_scorer`` value that selects this scorer (registry key).
    name: str
    #: Forward tensors the scorer reads; ``"qk"`` for every gate-free scorer.
    consumes: str = "qk"
    #: Settings only this scorer understands, declared rather than wired
    #: through configuration (see scorer_options.py). The declaration owns each
    #: setting's default, accepted values and help text; the factory resolves
    #: them and passes them to ``__init__`` as keyword arguments of the declared
    #: name. A scorer with no settings leaves this empty.
    OPTIONS: tuple[ScorerOption, ...] = ()

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        module: nn.Module | None = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """Return per-KV-head scores ``[num_kv_heads, T]`` (float32); higher =
        more important. ``value`` / ``module`` / ``position_offset`` are part of
        the shared contract — a scorer that does not need them ``del``s them."""
        ...

    # --- Optional: scoring the whole cache, not just the fresh chunk ---------
    #
    # ``forward`` scores one chunk as it is written, which is all a chunk-local
    # eviction target needs. A FIXED KV BUDGET needs more: every live position
    # competes at every eviction, so a position written many chunks ago must be
    # scorable now. Two kinds of scorer can satisfy that, and they differ in
    # what has to be remembered:
    #
    # * A score that is a function of the cached KEYS alone (KeyDiff: similarity
    #   to the mean direction of the keys in the cache) can be RECOMPUTED at
    #   every eviction from the cache itself — nothing needs to be stored, and
    #   the score is the paper's, not an approximation of it. Such a scorer sets
    #   ``rescores_cache`` and implements ``score_cached``.
    # * A score that depends on the history of queries (H2O: attention mass
    #   accumulated over every query so far) cannot be recovered from the cache,
    #   and is instead accumulated into a per-position buffer. That is a separate
    #   mechanism (see ``slot_scores.py``), not this method.
    #
    # A scorer that does neither keeps the score its chunk produced, which is an
    # approximation under a budget: scores from different chunks were computed
    # against different reference quantities, so ranking them together has no
    # common scale.

    #: Whether the scorer can rescore already-cached positions (``score_cached``
    #: implemented). Read by ``slot_scores`` to pick the score source under a
    #: fixed budget.
    rescores_cache: bool = False
    #: What ``score_cached`` needs materialised from the cache, as a subset of
    #: ``RESCORE_INPUTS``. The runner builds exactly these and nothing else, so
    #: a key-only method (KeyDiff) pays for one read while a method that also
    #: needs values or positions can be added WITHOUT changing this contract
    #: again. Empty unless ``rescores_cache`` is set.
    rescore_inputs: tuple[str, ...] = ()

    def score_cached(self, cached: "CachedPositions") -> torch.Tensor:
        """Score every live position of ONE head group from the cache.

        Args:
            cached: the inputs this scorer declared in ``rescore_inputs``,
                covering one head group's live slots in slot order.

        Returns:
            ``[page_group_size, num_positions]`` float32 scores, higher = more
            important, on the same device as the inputs.

        Only called when ``rescores_cache`` is set; the base raises so a scorer
        that advertises the capability without implementing it fails loudly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets rescores_cache but does not implement "
            "score_cached.")


#: Everything a rescoring scorer may ask the runner to materialise. Names are
#: declared here rather than as bare strings at each site so a scorer, the
#: runner that builds them and the tests agree on one vocabulary.
RESCORE_KEYS = "keys"
RESCORE_VALUES = "values"
RESCORE_POSITIONS = "positions"
RESCORE_INPUTS: tuple[str, ...] = (
    RESCORE_KEYS, RESCORE_VALUES, RESCORE_POSITIONS)


@dataclass(frozen=True)
class CachedPositions:
    """One head group's live cache slots, as a rescoring scorer sees them.

    Only the fields the scorer declared in ``rescore_inputs`` are populated;
    the rest are ``None``, so reading an undeclared field is a mistake that
    surfaces immediately rather than a silently wrong score.

    Slot order, not sequence order: eviction compacts survivors towards the
    front, so slot ``i`` holds whatever token survived into it. That is why
    ``positions`` exists as a separate field — after the first eviction a
    slot's global sequence position can no longer be inferred from its index.
    """
    #: ``[page_group_size, num_positions, head_size]`` post-RoPE keys, one row
    #: per KV head (cluster column).
    keys: torch.Tensor | None = None
    #: ``[page_group_size, num_positions, head_size]`` values, same layout.
    values: torch.Tensor | None = None
    #: ``[num_positions]`` global sequence position of each live slot.
    positions: torch.Tensor | None = None
    #: Live slots in this group; the trailing dimension of every field above.
    num_positions: int = 0

    def require(self, field: str) -> torch.Tensor:
        """Return an input the scorer declared, or say which declaration is
        missing. Scorers use this instead of asserting on ``None`` so the error
        names the fix (``rescore_inputs``) rather than the symptom."""
        value = getattr(self, field, None)
        if value is None:
            raise RuntimeError(
                f"cached {field} were not materialised; a scorer that reads "
                f"them must list {field!r} in rescore_inputs.")
        return value
