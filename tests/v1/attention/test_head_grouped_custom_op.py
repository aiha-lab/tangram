# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``_head_grouped_attention_impl`` (the body of the
``vllm::unified_attention_head_grouped`` custom op).

These run on CPU with a mock ``impl.forward``: they prove that the op body
hands the backend exactly the same member-major tensors and per-layer
metadata — and produces the same output rows — whether or not Q/K/V/output
are padded past ``attn_metadata.num_actual_tokens``. Padding happens whenever
piecewise CUDA graphs round the batch up to a capture size, so this is the
invariant that keeps graph mode numerically identical to eager.
"""
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from vllm.attention.layer import _head_grouped_attention_impl
from vllm.v1.attention.backends.head_grouped_layout import (
    identity_member_clusters,
    identity_member_columns,
)

NUM_REQS = 3
NUM_KV_HEADS = 4
NUM_QUERY_HEADS_PER_KV = 2
NUM_HEADS = NUM_KV_HEADS * NUM_QUERY_HEADS_PER_KV
HEAD_SIZE = 8
HIDDEN = NUM_HEADS * HEAD_SIZE
PAGE_GROUP_SIZE = 2
NUM_LAYERS = 2
LAYER_IDX = 1
MAX_BLOCKS = 5
PADDED_TOKENS = 8  # capture-size padding applied on top of the actual tokens


@dataclasses.dataclass
class _FakeMetadata:
    """Just the fields ``_head_grouped_attention_impl`` reads.

    A dataclass because the member-major path rebuilds per-layer metadata via
    ``dataclasses.replace``.
    """

    num_actual_tokens: int
    num_head_groups_per_layer: int
    page_group_size: int
    head_grouped_decode_layout: bool
    per_layer_md: list | None = None
    cluster_block_table: torch.Tensor | None = None
    clusters_per_layer: torch.Tensor | None = None
    cols_per_layer: torch.Tensor | None = None
    seq_lens_grouped: torch.Tensor | None = None
    slot_mapping_grouped: torch.Tensor | None = None
    query_start_loc_grouped: torch.Tensor | None = None
    # Overwritten per layer by ``dataclasses.replace`` on the member-major
    # path; never read before being replaced.
    block_table: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None


class _RecordingImpl:
    """Mock backend: records its inputs and writes ``2 * q`` into output.

    Writing a deterministic function of the query lets the test verify the
    op body's inverse (member-major → token-major) reshape independently:
    whatever permutation the body applies to q on the way in must be undone
    on the way out, so the final token-major output must equal ``2 * q``.
    """

    def __init__(self):
        self.calls = []

    def forward(self, layer, q, k, v, kv_cache, md, output=None):
        self.calls.append(
            SimpleNamespace(
                q=q.clone(), k=k.clone(), v=v.clone(),
                block_table=md.block_table.clone(),
                seq_lens=md.seq_lens.clone(),
                slot_mapping=md.slot_mapping.clone(),
                query_start_loc=md.query_start_loc.clone(),
                num_actual_tokens=md.num_actual_tokens,
            )
        )
        output.copy_(2.0 * q)


def _make_layer(impl) -> SimpleNamespace:
    return SimpleNamespace(
        use_output=True,
        num_groups_per_layer=NUM_KV_HEADS // PAGE_GROUP_SIZE,
        page_group_size=PAGE_GROUP_SIZE,
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        layer_idx=LAYER_IDX,
        impl=impl,
    )


def _qkv(num_tokens: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(num_tokens, NUM_HEADS * HEAD_SIZE, generator=gen)
    k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE, generator=gen)
    v = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE, generator=gen)
    return q, k, v


def _pad(x: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Append garbage rows, as capture-size padding leaves them undefined."""
    pad_rows = torch.full(
        (num_tokens - x.shape[0], x.shape[1]), float("nan"), dtype=x.dtype
    )
    return torch.cat([x, pad_rows], dim=0)


def _decode_metadata() -> _FakeMetadata:
    """Decode layout: one per-layer overlay per layer, each already sized to
    ``num_reqs * num_kv_heads`` virtual sequences (what the builder emits)."""
    num_virtual_seqs = NUM_REQS * NUM_KV_HEADS
    per_layer_md = []
    for layer_idx in range(NUM_LAYERS):
        per_layer_md.append(
            SimpleNamespace(
                num_actual_tokens=num_virtual_seqs,
                block_table=torch.arange(
                    (layer_idx + 1) * num_virtual_seqs * MAX_BLOCKS,
                    dtype=torch.int32,
                )[-num_virtual_seqs * MAX_BLOCKS:].reshape(
                    num_virtual_seqs, MAX_BLOCKS),
                seq_lens=torch.randint(
                    1, 64, (num_virtual_seqs,),
                    generator=torch.Generator().manual_seed(layer_idx)),
                slot_mapping=torch.arange(num_virtual_seqs,
                                          dtype=torch.int64),
                query_start_loc=torch.arange(num_virtual_seqs + 1,
                                             dtype=torch.int32),
            )
        )
    return _FakeMetadata(
        num_actual_tokens=NUM_REQS,
        num_head_groups_per_layer=NUM_KV_HEADS // PAGE_GROUP_SIZE,
        page_group_size=PAGE_GROUP_SIZE,
        head_grouped_decode_layout=True,
        per_layer_md=per_layer_md,
    )


def _member_major_metadata(num_actual_tokens: int) -> _FakeMetadata:
    """Prefill/mixed layout with identity member↔cluster maps."""
    num_clusters_per_layer = NUM_KV_HEADS // PAGE_GROUP_SIZE
    num_clusters_total = NUM_LAYERS * num_clusters_per_layer
    num_members_total = NUM_LAYERS * NUM_KV_HEADS
    gen = torch.Generator().manual_seed(7)
    cluster_block_table = torch.randint(
        0, 100, (NUM_REQS, num_clusters_total, MAX_BLOCKS),
        dtype=torch.int32, generator=gen)
    clusters = identity_member_clusters(num_members_total, PAGE_GROUP_SIZE)
    cols = identity_member_columns(num_members_total, PAGE_GROUP_SIZE)
    seq_lens_grouped = torch.randint(
        1, 64, (num_members_total, NUM_REQS), generator=gen)
    slot_mapping_grouped = torch.randint(
        0, 10_000, (num_members_total, num_actual_tokens),
        dtype=torch.int64, generator=gen)
    # cu_seqlens over (kv_head, req) member sequences: kv head h of request r
    # owns query rows [h * num_actual_tokens + qsl[r], h * num_actual_tokens
    # + qsl[r + 1]) — matching the builder's arange(kvh) * num_actual_tokens
    # offsets.
    query_start_loc = torch.tensor(
        [0, 1, 2, num_actual_tokens][:NUM_REQS + 1], dtype=torch.int32)
    offsets = (
        torch.arange(NUM_KV_HEADS, dtype=torch.int32) * num_actual_tokens
    )
    query_start_loc_grouped = torch.cat(
        [
            (query_start_loc[:NUM_REQS] + offsets[:, None]).reshape(-1),
            torch.tensor([NUM_KV_HEADS * num_actual_tokens],
                         dtype=torch.int32),
        ]
    )
    return _FakeMetadata(
        num_actual_tokens=num_actual_tokens,
        num_head_groups_per_layer=num_clusters_per_layer,
        page_group_size=PAGE_GROUP_SIZE,
        head_grouped_decode_layout=False,
        cluster_block_table=cluster_block_table,
        clusters_per_layer=clusters.view(NUM_LAYERS, NUM_KV_HEADS),
        cols_per_layer=cols.view(NUM_LAYERS, NUM_KV_HEADS),
        seq_lens_grouped=seq_lens_grouped,
        slot_mapping_grouped=slot_mapping_grouped,
        query_start_loc_grouped=query_start_loc_grouped,
    )


def _run(metadata: _FakeMetadata, num_tokens: int, num_actual: int):
    """Run the impl with inputs of ``num_tokens`` rows (>= num_actual)."""
    impl = _RecordingImpl()
    layer = _make_layer(impl)
    q, k, v = _qkv(num_actual)
    if num_tokens > num_actual:
        q, k, v = _pad(q, num_tokens), _pad(k, num_tokens), _pad(v, num_tokens)
    output = torch.full((num_tokens, HIDDEN), float("nan"))
    kv_cache = torch.empty(0)
    _head_grouped_attention_impl(layer, q, k, v, output, metadata, kv_cache)
    assert len(impl.calls) == 1
    return impl.calls[0], output, q


@pytest.mark.parametrize("layout", ["decode", "member_major"])
def test_padded_matches_unpadded(layout: str):
    """Backend inputs and actual output rows must not change when the batch
    is padded to a CUDA-graph capture size."""
    if layout == "decode":
        metadata = _decode_metadata()
    else:
        metadata = _member_major_metadata(NUM_REQS)

    call_exact, out_exact, _ = _run(metadata, NUM_REQS, NUM_REQS)
    call_padded, out_padded, _ = _run(metadata, PADDED_TOKENS, NUM_REQS)

    for field in ("q", "k", "v", "block_table", "seq_lens", "slot_mapping",
                  "query_start_loc"):
        exact = getattr(call_exact, field)
        padded = getattr(call_padded, field)
        assert torch.equal(exact, padded), (
            f"{layout}: backend received different `{field}` under padding")
    assert call_exact.num_actual_tokens == call_padded.num_actual_tokens

    assert torch.equal(out_exact[:NUM_REQS], out_padded[:NUM_REQS]), (
        f"{layout}: actual output rows differ under padding")


@pytest.mark.parametrize("layout", ["decode", "member_major"])
@pytest.mark.parametrize("num_tokens", [NUM_REQS, PADDED_TOKENS])
def test_inverse_reshape_roundtrip(layout: str, num_tokens: int):
    """With the backend writing ``2 * q``, the member-major permutation on the
    way in must be exactly undone on the way out: token-major output rows must
    equal ``2 * q`` for every actual token."""
    if layout == "decode":
        metadata = _decode_metadata()
    else:
        metadata = _member_major_metadata(NUM_REQS)

    _, output, q = _run(metadata, num_tokens, NUM_REQS)
    assert torch.equal(output[:NUM_REQS], 2.0 * q[:NUM_REQS])


def test_member_major_layer_slice():
    """The backend must see layer ``LAYER_IDX``'s slice of the member-major
    precomputes, with the virtual block id folding in the member's column."""
    metadata = _member_major_metadata(NUM_REQS)
    call, _, _ = _run(metadata, NUM_REQS, NUM_REQS)

    layer_start = LAYER_IDX * NUM_KV_HEADS
    layer_end = layer_start + NUM_KV_HEADS
    assert torch.equal(
        call.seq_lens,
        metadata.seq_lens_grouped[layer_start:layer_end].reshape(-1))
    assert torch.equal(
        call.slot_mapping,
        metadata.slot_mapping_grouped[layer_start:layer_end].reshape(-1))
    assert call.num_actual_tokens == NUM_KV_HEADS * NUM_REQS

    # Independent virtual-block-table reconstruction for one (member, req):
    # member m of this layer reads cluster ``clusters_per_layer[L, m]`` and
    # folds its column: ``physical * page_group_size + column``.
    member = 3
    req = 1
    cluster = metadata.clusters_per_layer[LAYER_IDX, member].item()
    col = metadata.cols_per_layer[LAYER_IDX, member].item()
    expected_row = (
        metadata.cluster_block_table[req, cluster].to(torch.int64)
        * PAGE_GROUP_SIZE + col
    )
    # Backend row order is member-major reshaped to (kv_head, req):
    # row = member * num_reqs + req after the permute in the op body.
    got_row = call.block_table[member * NUM_REQS + req].to(torch.int64)
    assert torch.equal(got_row, expected_row)


def test_none_metadata_zero_fills_output():
    """Profile and compilation dummy runs pass no metadata; the op must zero
    its (in-place) output instead of leaving ``torch.empty`` garbage."""
    impl = _RecordingImpl()
    layer = _make_layer(impl)
    q, k, v = _qkv(PADDED_TOKENS)
    output = torch.full((PADDED_TOKENS, HIDDEN), float("nan"))
    _head_grouped_attention_impl(
        layer, q, k, v, output, None, torch.empty(0))
    assert impl.calls == []
    assert torch.equal(output, torch.zeros_like(output))
