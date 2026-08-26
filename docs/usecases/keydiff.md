# KeyDiff use case

Source: [`vllm/v1/attention/compression/keydiff.py`](../../vllm/v1/attention/compression/keydiff.py) ·
Paper: [arXiv:2504.15364](https://arxiv.org/abs/2504.15364)

## How it works

The scorer reads the model's post-RoPE keys and nothing else:

```python
k = key.reshape(chunk_len, self.num_kv_heads, self.head_size).float()
anchor = k.mean(dim=0, keepdim=True)             # mean key of the chunk, mu(K)
score = -F.cosine_similarity(k, anchor, dim=-1)  # [T, H], higher = keep
```

Keys pointing near the chunk's mean direction are the least distinctive, so negating the
cosine similarity ranks the redundant tokens lowest and evicts them first. Two
consequences follow: KeyDiff is **gate-free** (no checkpoint to load, unlike FastKVzip)
and **query-independent** (it never reads queries, so chunked prefill carries no extra
state across chunks).

The `[num_kv_heads, chunk_len]` score it returns is the contract every Tangram scorer
shares, so `compression_budget_scope` stays an orthogonal knob.

### The anchor has two published spellings

`--compression-scorer-options anchor=...` selects which mean the keys are compared
against:

| value | anchor | where it comes from |
|---|---|---|
| `unnormalized` (default) | mean of the raw keys, `mu(K)` | the paper's §3.2: *"We evaluate the efficient KeyDiff described in Figure 3 using unnormalized keys k in all subsequent sections"* — what its reported numbers use |
| `normalized` | mean of the L2-normalized directions, `mu(K-hat)` | Eq. (8) as written; also what NVIDIA KVpress computes |

The paper reports the two as equally accurate (Table 15). `cosine_similarity`
normalizes both of its arguments, so only the anchor's *direction* differs: averaging
raw keys lets a long key pull the mean towards itself, averaging directions gives every
key the same pull.

### Under a fixed KV budget the anchor spans the whole cache

With `--compression-budget-tokens` nothing is locked in, so every live position competes
at each eviction and KeyDiff rescores them all against the mean of the keys **currently
cached** — Eq. (8)'s `K`, not one chunk's. That is the paper's rule (Eq. 4:
`C <- [K || k_new]; C' <- pi_N(C)`), and it needs no stored statistics because the keys
it depends on are already in the cache. The selected `anchor` applies there too.

## How to run

Speedup:

```bash
cd benchmarks/tangram/speedup
SCORERS=keydiff RATIOS="0.0 0.5 0.75 0.9" ./run_speedup.sh
```

SCBench accuracy:

```bash
cd benchmarks/tangram
SCORER=keydiff SCOPE=layer DATASET=mid RATIOS="0.0 0.5 0.75 0.9" \
bash benchmark_scbench.sh
```

Override `MODEL=` for another model, and `DATASET=` for another task group
(`short` / `mid` / `long` / `multi`).

### Knobs

`SCORER` (or `SCORERS` in `run_speedup.sh`, which takes a list) picks the importance
scorer, and **it must be `keydiff` for anything on this page to apply** — the scripts
otherwise default to `snapkv`, and `run_speedup.sh` to `snapkv fastkvzip`.

`SCOPE` is the *budget scope*: the range a KV budget is shared over, which decides
whether heads may keep different numbers of tokens.

| `SCOPE` | Budget scope |
| ------- | ------------ |
| `uniform` | Every (layer, head group) keeps the same token count; only *which* tokens are kept differs. Needs no cluster map. |
| `layer` | Each layer gets an equal budget, spread non-uniformly across the heads in that layer. Needs the `_perlayer` cluster map. |
| `global` | One global budget spread across all layers and heads, so an important head in any layer can keep more. Needs the cross-layer cluster map. |

KeyDiff
cluster maps for every verified model already ship under
[`tools/head_group_clustering/cluster_maps/keydiff/`](../../tools/head_group_clustering/cluster_maps/keydiff),
so the `layer` / `global` scopes work with no extra step.

## Speedup

<p align="center">
  <img src="../assets/speedup/speedup_keydiff.png" alt="Tangram end-to-end speedup, KeyDiff scorer" width="100%"/>
</p>

## Accuracy

Every point below is the mean over **all 13 SCBench datasets** — the whole benchmark,
across its `short`, `mid`, `long` and `multi` task groups — not a single task and not one
group. The `DATASET=mid` command above runs one group, so reproducing a bar means
sweeping all four and averaging over the datasets they cover.

Five models, KeyDiff scorer, w/ against w/o Tangram at each compression ratio. The dashed
line is the full-KV reference (itself per-method, since a scorer's observation window
applies even at full budget) and the violet series is the gap between the two bars, as a
percentage of the w/o-Tangram score, on the right axis.

<p align="center">
  <img src="../assets/accuracy/accuracy_scbench_keydiff.png" alt="SCBench accuracy, KeyDiff scorer, Tangram vs PyTorch across five models" width="100%"/>
</p>
