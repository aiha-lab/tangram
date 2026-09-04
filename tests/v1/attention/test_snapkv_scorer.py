# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SnapKV scoring must equal the reference formula bit for bit.

The scorer folds the GQA group into the matmul's row axis so K is read once
instead of being broadcast-copied per query head. The reference below is the
formula as the paper states it -- broadcast over the group, then ``amax`` --
and the two must agree exactly, on CPU and on a GPU when one is present.

    python -m pytest --noconftest -q tests/v1/attention/test_snapkv_scorer.py
"""
import math

import pytest
import torch
import torch.nn.functional as F

from vllm.v1.attention.compression.snapkv import SnapKVScorer

NUM_KV_HEADS = 4
NUM_Q_PER_KV = 3
HEAD = 16
WINDOW = 8
KERNEL = 5


def reference(query, key, *, window):
    """Broadcast form: ``q[kv, g, w, d] @ k_t[kv, 1, d, T]``."""
    chunk_len = query.shape[0]
    q = query.reshape(chunk_len, NUM_KV_HEADS, NUM_Q_PER_KV, HEAD)
    k = key.reshape(chunk_len, NUM_KV_HEADS, HEAD)
    w = window if chunk_len >= 1000 else min(16, chunk_len)
    q = q[chunk_len - w:].permute(1, 2, 0, 3)
    k_t = k.permute(1, 2, 0)
    attn = torch.matmul(q, k_t.unsqueeze(1)) / math.sqrt(HEAD)
    attn = attn.amax(dim=1)
    weights = torch.softmax(attn, dim=-1, dtype=torch.float32).mean(dim=-2)
    return F.max_pool1d(weights, kernel_size=KERNEL, padding=KERNEL // 2,
                        stride=1)


def make(chunk_len, dtype, device):
    torch.manual_seed(chunk_len)
    q = torch.randn(chunk_len, NUM_KV_HEADS * NUM_Q_PER_KV * HEAD,
                    dtype=dtype, device=device)
    k = torch.randn(chunk_len, NUM_KV_HEADS * HEAD, dtype=dtype, device=device)
    return q, k


@pytest.mark.parametrize("chunk_len", [5, 16, 300, 1200])
def test_matches_reference_on_cpu(chunk_len):
    scorer = SnapKVScorer(NUM_KV_HEADS, HEAD, NUM_Q_PER_KV, window=WINDOW,
                          kernel=KERNEL)
    q, k = make(chunk_len, torch.float32, "cpu")
    assert torch.equal(scorer(q, k), reference(q, k, window=WINDOW))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("chunk_len", [300, 1200, 8192])
def test_matches_reference_on_gpu_bf16(chunk_len):
    scorer = SnapKVScorer(NUM_KV_HEADS, HEAD, NUM_Q_PER_KV, window=WINDOW,
                          kernel=KERNEL)
    q, k = make(chunk_len, torch.bfloat16, "cuda")
    assert torch.equal(scorer(q, k), reference(q, k, window=WINDOW))
