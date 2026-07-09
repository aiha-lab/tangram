<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <img src="docs/assets/logos/tangram-logo-text.png" alt="Tangram" width="480"/>
</p>

<h3 align="center">
Tangram: Unlocking Non-Uniform KV Cache Compression for Efficient Multi-turn LLM Serving
</h3>

<p align="center">
| <a href="https://aiha-lab.github.io/tangram-page/"><b>Project Page</b></a> | <a href="https://aiha-lab.github.io/tangram-page/#"><b>Paper</b></a> |
</p>

<p align="center">
  <img src="docs/assets/speedup/speedup.png" alt="Tangram end-to-end speedup vs vLLM 0.11.1" width="100%"/>
</p>

**Tangram** is a serving system that makes non-uniform KV cache compression
practical for multi-turn LLM serving, built on top of
[vLLM](https://github.com/vllm-project/vllm).

**Highlights**

- **KV Cache Compression on vLLM with Ragged Paging** — non-uniform and uniform KV cache compression, natively integrated into vLLM
- **Seamless vLLM integration** — fully compatible with paged attention, continuous batching, chunked prefill, and CUDA graph mode
- **Real memory reclamation** — compressed KV cache is actually freed, turning memory savings into higher serving throughput
- **Zero runtime scheduling overhead** — budget reservation and ahead-of-time (AOT) load balancing keep compression off the critical path

## Installation

```bash
git clone https://github.com/aiha-lab/tangram.git
cd tangram
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

## Quickstart

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B-Instruct-2507",
    compression_ratio=0.5,                  # keep 50% of the KV cache (1.0 = no compression)
    compression_scorer="snapkv",            # snapkv | keydiff | expected_attention | fastkvzip
    compression_level="crosslayer_cluster", # crosslayer_cluster | perlayer_cluster | uniform
)

out = llm.generate(["What is KV cache compression?"], SamplingParams(max_tokens=128))
print(out[0].outputs[0].text)
```

## Configuration

| Config | Description |
| ------ | ----------- |
| `compression_ratio` | KV retention fraction; `1.0` = FullKV (no compression). |
| `compression_scorer` | Importance scorer: `snapkv` \| `keydiff` \| `expected_attention` \| `fastkvzip`. |
| `compression_level` | KV-budget scope — see below. |
| `page_group_size` | Heads per page (H<sub>p</sub>): attention heads managed together in one KV-cache page; they share one paged budget. |

**`compression_level` options**

- `crosslayer_cluster` — non-uniform; a single global KV budget is distributed across all layers and heads, so important heads in any layer can keep more tokens.
- `perlayer_cluster` — non-uniform; each layer gets an equal KV budget, distributed non-uniformly across the heads within that layer.
- `uniform` — every attention head keeps the same number of tokens (`ratio × seq_len`); only *which* tokens are kept differs per head.

## Supported Compression

**✅ Supported**

- SnapKV ([paper](https://arxiv.org/abs/2404.14469))
- KeyDiff ([paper](https://arxiv.org/abs/2504.15364))
- ExpectedAttention ([paper](https://arxiv.org/abs/2510.00636))
- FastKVzip ([paper](https://arxiv.org/abs/2601.17668))

**🚧 WIP**

- TOVA
- PyramidKV

## Supported Models

The following models have been verified with Tangram. More models are on the way.

| Model | Checkpoint |
| ----- | ---------- |
| Qwen3 4B | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Llama 3.1 8B | [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Gemma 3 12B | [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) |
| GPT-OSS 20B | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) |
| Qwen3 30B (MoE)&nbsp;* | [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) |

\* Verified with `tensor_parallel_size=2`.

## Accuracy

[RULER](https://arxiv.org/abs/2404.06654) 8K

<details>
<summary><b>Non-uniform (<code>perlayer_cluster</code>), H<sub>p</sub> = 4</b></summary>

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">FullKV</th>
<th colspan="4">SnapKV</th>
<th colspan="4">KeyDiff</th>
<th colspan="4">ExpectedAttention</th>
<th colspan="4">FastKVzip</th>
</tr>
<tr>
<th>75%</th><th>50%</th><th>25%</th><th>10%</th>
<th>75%</th><th>50%</th><th>25%</th><th>10%</th>
<th>75%</th><th>50%</th><th>25%</th><th>10%</th>
<th>75%</th><th>50%</th><th>25%</th><th>10%</th>
</tr>
</thead>
<tbody>
<tr><td>qwen3-4b</td><td>93.8</td><td>89.0</td><td>80.7</td><td>70.0</td><td>59.2</td><td>92.3</td><td>84.1</td><td>74.6</td><td>61.0</td><td>93.5</td><td>93.2</td><td>80.4</td><td>60.9</td><td>93.5</td><td>92.7</td><td>83.0</td><td>39.6</td></tr>
<tr><td>llama3.1-8b</td><td>94.6</td><td>91.7</td><td>88.4</td><td>75.6</td><td>61.0</td><td>94.4</td><td>91.0</td><td>81.8</td><td>71.0</td><td>94.3</td><td>93.2</td><td>77.0</td><td>55.9</td><td>94.6</td><td>94.3</td><td>85.0</td><td>62.6</td></tr>
<tr><td>gemma3-12b</td><td>91.6</td><td>77.4</td><td>67.0</td><td>59.0</td><td>51.9</td><td>85.3</td><td>78.1</td><td>69.5</td><td>50.1</td><td>90.4</td><td>85.2</td><td>72.6</td><td>60.5</td><td>91.4</td><td>89.8</td><td>82.2</td><td>52.3</td></tr>
<tr><td>gptoss-20b</td><td>83.4</td><td>81.9</td><td>76.8</td><td>66.4</td><td>53.0</td><td>82.8</td><td>81.8</td><td>69.0</td><td>46.1</td><td>83.8</td><td>82.5</td><td>64.9</td><td>38.2</td><td>83.9</td><td>88.5</td><td>74.9</td><td>41.0</td></tr>
</tbody>
</table>

</details>

<details>
<summary><b>Uniform (<code>uniform</code>), H<sub>p</sub> = 4</b></summary>

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">FullKV</th>
<th colspan="4">SnapKV</th>
<th colspan="4">KeyDiff</th>
<th colspan="4">ExpectedAttention</th>
</tr>
<tr>
<th>75%</th><th>50%</th><th>25%</th><th>10%</th>
<th>75%</th><th>50%</th><th>25%</th><th>10%</th>
<th>75%</th><th>50%</th><th>25%</th><th>10%</th>
</tr>
</thead>
<tbody>
<tr><td>qwen3-4b</td><td>93.7</td><td>85.0</td><td>77.0</td><td>68.8</td><td>59.0</td><td>89.8</td><td>80.2</td><td>70.9</td><td>55.1</td><td>77.3</td><td>50.5</td><td>28.4</td><td>15.6</td></tr>
<tr><td>llama3.1-8b</td><td>93.1</td><td>89.0</td><td>84.2</td><td>71.2</td><td>56.0</td><td>87.8</td><td>83.2</td><td>76.1</td><td>67.7</td><td>79.3</td><td>61.4</td><td>40.3</td><td>21.0</td></tr>
<tr><td>gemma3-12b</td><td>91.7</td><td>76.2</td><td>66.5</td><td>58.7</td><td>51.7</td><td>86.9</td><td>80.3</td><td>71.2</td><td>50.3</td><td>63.7</td><td>46.1</td><td>29.2</td><td>21.4</td></tr>
<tr><td>gptoss-20b</td><td>83.7</td><td>81.9</td><td>77.0</td><td>66.2</td><td>53.2</td><td>79.1</td><td>71.0</td><td>57.0</td><td>33.8</td><td>74.8</td><td>48.9</td><td>25.9</td><td>13.4</td></tr>
<tr><td>qwen3-30b</td><td>95.3</td><td>87.5</td><td>80.7</td><td>74.3</td><td>64.7</td><td>88.7</td><td>82.0</td><td>74.4</td><td>66.9</td><td>60.6</td><td>40.5</td><td>29.0</td><td>18.5</td></tr>
</tbody>
</table>

</details>

## Benchmarks

### RULER

Run one compression method at one ratio on RULER 8K:

```bash
cd benchmarks/tangram

MODEL=meta-llama/Llama-3.1-8B-Instruct \
SCORER=snapkv LEVEL=crosslayer_cluster RATIOS=0.5 LENGTHS=8192 \
bash benchmark_ruler.sh
```

- `SCORER` — `snapkv` | `keydiff` | `expected_attention`
- `RATIOS` — KV retention fraction (`1.0` = FullKV reference)
- `LEVEL` — selection level; see [Configuration](#configuration)

### Speedup

Measure end-to-end generation speedup — wall-clock at `r=1.0` (uncompressed)
vs compressed ratios on an SCBench task:

```bash
cd benchmarks/tangram/speedup
./run_speedup.sh
```

## Citation

If you use Tangram for your research, please cite our [paper](https://arxiv.org/abs/2606.06302):

```
@misc{kim2026tangramunlockingnonuniformkv,
      title={Tangram: Unlocking Non-Uniform KV Cache Compression for Efficient Multi-turn LLM Serving}, 
      author={Hyungmin Kim and Minsoo Kim and Hongseok Kim and Jungwook Choi},
      year={2026},
      eprint={2606.06302},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.06302}, 
}
```

## Acknowledgements

Tangram project is built on top of [vLLM](https://github.com/vllm-project/vllm), and its
compression scorer implementations are adapted from [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress).
