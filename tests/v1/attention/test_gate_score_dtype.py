# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The workspace reserves the dtype the gate actually emits.

A trained gate checkpoint keeps ``k_base`` and both RMSNorm weights in float32
while the projections are in the model dtype, so a bfloat16 model's gate still
scores in float32. Rounding those scores to bfloat16 would change the keep
decision, so the workspace must hold them as they are.
"""
import torch

from vllm.v1.attention.compression.gate import CompressionGate
from vllm.v1.attention.compression.workspace import WorkspaceSpec

MODEL_DTYPE = torch.bfloat16
HIDDEN_DIM = 32
NUM_KV_HEADS = 2


def test_workspace_holds_the_gate_score_dtype():
    gate = CompressionGate(
        layer_idx=0,
        hidden_dim=HIDDEN_DIM,
        output_dim=16,
        num_kv_heads=NUM_KV_HEADS,
        num_groups=4,
        sink_dim=16,
        dtype=MODEL_DTYPE,
    )
    score = gate(torch.randn(8, HIDDEN_DIM, dtype=MODEL_DTYPE))

    spec = WorkspaceSpec.from_config(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        num_groups=NUM_KV_HEADS,
        page_group_size=1,
        max_num_reqs=4,
        max_model_len=1 << 20,
        model_dtype=MODEL_DTYPE,
        chunk_size=32,
        window_size=8,
        n_sink_tokens=4,
        budget_tokens=None,
        evict_current_chunk=False,
        scorer="fastkvzip",
        slot_score_source="auto",
    )

    assert spec.score_dtype == score.dtype
