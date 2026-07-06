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
