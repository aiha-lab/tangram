# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which decoder layers are compressible, and which hold a sliding window.

One scan of the model's ``Attention`` modules answers both, and the answer has
to be one answer: the two sets are complements, and every consumer must agree
about the boundary or KV is freed for a layer that still reads it.

The signal is a layer's ``sliding_window``. Unset means full attention, which
is what FastKVZip compresses and therefore what a head-group cluster map is
authored over. Set means the layer keeps full KV within its window, so instead
of compressing it the engine returns its out-of-window front blocks to the
pool.

Both consumers -- the compression engine's startup
(``CompressionModelRunnerMixin._init_compression``) and the ragged
FlashAttention metadata builder -- come through here, so their layer sets cannot
drift apart.
"""
from dataclasses import dataclass

from vllm.config.vllm import VllmConfig


@dataclass(frozen=True)
class AttentionLayerSplit:
    """Physical decoder-layer indices, ascending, and the sliding window.

    ``full`` and ``sliding`` partition the model's layers, so a dense model has
    every layer in ``full``, an empty ``sliding``, and ``sliding_window`` of 0.
    """

    full: list[int]
    sliding: list[int]
    sliding_window: int


def split_attention_layers(vllm_config: VllmConfig) -> AttentionLayerSplit:
    """Partition the model's decoder layers by whether they slide.

    A single window across all sliding-window layers is assumed (gemma-3 uses
    1024); a model that mixes window sizes raises ``NotImplementedError``
    rather than picking one, because the eviction arithmetic takes one window
    for every sliding layer.

    Modules whose name carries no decoder layer index -- encoder-only
    attention, for instance -- are skipped by both halves.
    """
    # Imported here, not at module scope: vllm.attention.layer imports this
    # package, so a top-level import would close the cycle.
    from vllm.attention.layer import Attention
    from vllm.model_executor.models.utils import extract_layer_index
    from vllm.v1.attention.backends.utils import get_layers_from_vllm_config

    full: list[int] = []
    sliding: list[int] = []
    windows: set[int] = set()

    for layer_name, module in get_layers_from_vllm_config(
        vllm_config, Attention
    ).items():
        try:
            layer_index = extract_layer_index(layer_name)
        except (AssertionError, ValueError):
            continue

        window = getattr(module, "sliding_window", None)
        if window is None:
            full.append(layer_index)
            continue

        sliding.append(layer_index)
        windows.add(int(window))

    if len(windows) > 1:
        raise NotImplementedError(
            "Sliding-window KV eviction assumes a single window across the "
            f"sliding-window layers; got {sorted(windows)}.")

    return AttentionLayerSplit(
        full=sorted(full),
        sliding=sorted(sliding),
        sliding_window=int(next(iter(windows))) if windows else 0,
    )
