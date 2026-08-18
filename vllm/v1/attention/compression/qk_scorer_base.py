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

import torch
from torch import nn


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
    #   ``rescores_cache`` and implements ``score_cached_keys``.
    # * A score that depends on the history of queries (H2O: attention mass
    #   accumulated over every query so far) cannot be recovered from the cache,
    #   and is instead accumulated into a per-position buffer. That is a separate
    #   mechanism (see ``slot_scores.py``), not this method.
    #
    # A scorer that does neither keeps the score its chunk produced, which is an
    # approximation under a budget: scores from different chunks were computed
    # against different reference quantities, so ranking them together has no
    # common scale.

    #: Whether the scorer can rescore already-cached positions from the keys
    #: alone (``score_cached_keys`` implemented). Read by ``slot_scores`` to pick
    #: the score source under a fixed budget.
    rescores_cache: bool = False

    def score_cached_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """Score every live position of ONE head group from its cached keys.

        Args:
            keys: ``[page_group_size, num_positions, head_size]`` post-RoPE keys
                of the group's live cache slots, one row per KV head (cluster
                column), in slot order.

        Returns:
            ``[page_group_size, num_positions]`` float32 scores, higher = more
            important, on the same device as ``keys``.

        Only called when ``rescores_cache`` is set; the base raises so a scorer
        that advertises the capability without implementing it fails loudly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets rescores_cache but does not implement "
            "score_cached_keys.")
