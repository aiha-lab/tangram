# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base contract for gate-free query/key compression scorers (axis 2).

A ``QKScorer`` is a stateless ``nn.Module`` shared across every compressible
layer, turning one request-chunk's post-RoPE query/key (and optionally value)
into ``[num_kv_heads, T]`` importance scores, higher = keep. Declared once here
so the axis-2 registry can key off ``name`` and ``attach_scorers`` can rely on
``name`` / ``consumes`` existing.

FastKVZip is deliberately not a ``QKScorer``: it is checkpoint-backed and
consumes ``hidden_states``, so it loads and delivers on a separate path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from vllm.v1.attention.compression.scorer_options import ScorerOption


class QKScorer(nn.Module, ABC):
    """Gate-free query/key importance scorer (axis 2).

    A subclass sets ``name``, the ``compression_scorer`` value that selects it,
    and implements ``forward``. ``consumes`` names the tensors it reads;
    ``"qk"`` is the only value a gate-free scorer uses, so it is the default and
    a subclass need not repeat it.
    """

    #: ``compression_scorer`` value that selects this scorer (registry key).
    name: str
    #: Forward tensors the scorer reads; ``"qk"`` for every gate-free scorer.
    consumes: str = "qk"
    #: Settings only this scorer understands, declared rather than wired
    #: through configuration: the declaration owns each default, accepted value
    #: set and help text, and the factory resolves them into keyword arguments.
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
    # ``forward`` scores a chunk as it is written, but under a fixed budget an
    # old position competes again and must be scorable now. A scorer whose score
    # is a function of the cached keys sets ``rescores_cache`` and implements
    # ``score_cached``; ``slot_scores.py`` covers the rest.

    #: Whether ``score_cached`` is implemented, so already-cached positions can
    #: be rescored. Read by ``slot_scores`` to pick the source under a budget.
    rescores_cache: bool = False
    #: What ``score_cached`` needs materialised, a subset of
    #: ``RESCORE_INPUTS``. The runner builds exactly these, so a key-only method
    #: pays for one read and one needing values changes no contract.
    rescore_inputs: tuple[str, ...] = ()

    def score_cached(self, cached: "CachedPositions") -> torch.Tensor:
        """Score every live position of ONE head group from the cache.

        ``cached`` holds the inputs this scorer declared in ``rescore_inputs``,
        in slot order. Returns ``[page_group_size, num_positions]`` float32,
        higher = keep, on the inputs' device.

        Called only when ``rescores_cache`` is set; the base raises so a scorer
        advertising the capability without implementing it fails loudly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets rescores_cache but does not implement "
            "score_cached.")


#: Everything a rescoring scorer may ask the runner to materialise, named here
#: rather than as bare strings so scorer, runner and tests share one vocabulary.
RESCORE_KEYS = "keys"
RESCORE_VALUES = "values"
RESCORE_INPUTS: tuple[str, ...] = (RESCORE_KEYS, RESCORE_VALUES)


@dataclass(frozen=True)
class CachedPositions:
    """One head group's live cache slots, as a rescoring scorer sees them.

    Only the fields declared in ``rescore_inputs`` are populated and the rest
    are ``None``, so reading an undeclared one fails immediately instead of
    scoring wrongly.

    Slot order, NOT sequence order: eviction compacts survivors forward, so a
    slot's global sequence position cannot be inferred from its index.
    """
    #: ``[page_group_size, num_positions, head_size]`` post-RoPE keys, one row
    #: per KV head (cluster column).
    keys: torch.Tensor | None = None
    #: ``[page_group_size, num_positions, head_size]`` values, same layout.
    values: torch.Tensor | None = None
    #: Live slots in this group; the trailing dimension of every field above.
    num_positions: int = 0

    def require(self, field: str) -> torch.Tensor:
        """Return an input the scorer declared, or say which declaration is
        missing. Scorers use this instead of asserting on ``None`` so the error
        names the fix (``rescore_inputs``) rather than the symptom."""
        value = getattr(self, field)
        if value is None:
            raise RuntimeError(
                f"cached {field} were not materialised; a scorer that reads "
                f"them must list {field!r} in rescore_inputs.")
        return value
