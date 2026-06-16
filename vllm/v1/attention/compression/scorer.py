# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate-free query/key scorer factory (compression axis 2).

The single dispatch point mapping a ``compression_scorer`` name to its
gate-free scorer module (SnapKV, KeyDiff, …). Mirrors ``make_selection_level``
for axis 1: adding a scorer = one new module + one entry here, with no scorer
branching leaking elsewhere (the runner and compressor stay scorer-agnostic).

FastKVZip is deliberately NOT here — it is checkpoint-backed and consumes
hidden_states, loaded by ``KVCompressor.load_gate_checkpoint``. This factory
covers only the gate-free scorers that consume post-RoPE query/key.
"""
from __future__ import annotations

from torch import nn

from vllm.v1.attention.compression.keydiff import KeyDiffScorer
from vllm.v1.attention.compression.snapkv import SnapKVScorer
from vllm.v1.attention.compression.streamingllm import StreamingLLMScorer
from vllm.v1.attention.compression.expected_attention import (
    ExpectedAttentionScorer,
)
from vllm.v1.attention.compression.tova import TOVAScorer


def build_qk_scorer(
    name: str,
    *,
    num_kv_heads: int,
    num_q_per_kv: int,
    head_size: int,
    snap_window: int,
    snap_kernel: int,
    ea_use_covariance: bool = True,
    ea_use_vnorm: bool = True,
    ea_n_future_positions: int = 512,
    ea_epsilon: float = 1e-2,
) -> nn.Module:
    """Construct the gate-free query/key scorer selected by ``name``.

    ``num_q_per_kv`` is the per-rank GQA ratio (model q-heads / kv-heads);
    scorers that ignore it (KeyDiff) simply do not use it. Per-scorer
    hyperparameters (SnapKV's ``snap_window`` / ``snap_kernel``;
    ExpectedAttention's ``ea_*``) are likewise consumed only by the scorer that
    needs them. The returned module exposes ``consumes`` / ``name`` for the hook
    dispatch in ``attach_scorers``.
    """
    if name == "snapkv":
        return SnapKVScorer(
            num_kv_heads=num_kv_heads,
            num_q_per_kv=num_q_per_kv,
            head_size=head_size,
            snap_window=snap_window,
            snap_kernel=snap_kernel,
        )
    if name == "keydiff":
        return KeyDiffScorer(
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    if name == "streamingllm":
        return StreamingLLMScorer(num_kv_heads=num_kv_heads)
    if name == "tova":
        return TOVAScorer(
            num_kv_heads=num_kv_heads,
            num_q_per_kv=num_q_per_kv,
            head_size=head_size,
        )
    if name == "expected_attention":
        return ExpectedAttentionScorer(
            num_kv_heads=num_kv_heads,
            num_q_per_kv=num_q_per_kv,
            head_size=head_size,
            use_covariance=ea_use_covariance,
            use_vnorm=ea_use_vnorm,
            n_future_positions=ea_n_future_positions,
            epsilon=ea_epsilon,
        )
    raise ValueError(
        f"build_qk_scorer: unknown gate-free qk scorer {name!r}; expected "
        "'snapkv', 'keydiff', 'streamingllm', 'tova', or "
        "'expected_attention'.")
