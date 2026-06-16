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

**Tangram** is a serving system that makes non-uniform KV cache compression
practical for multi-turn LLM serving. It is built on top of
[vLLM](https://github.com/vllm-project/vllm).

**Highlights**

- **KV Cache Compression on vLLM** — brings both non-uniform (per-head) and uniform (per-layer) KV cache compression to vLLM serving
- **Seamless vLLM integration** — fully compatible with paged attention, continuous batching, and chunked prefill
- **Efficient LLM serving** — compressed KV cache is actually reclaimed, turning memory savings into higher serving throughput

**Core techniques**

1. **Budget Reservation** — static per-head memory footprint, no runtime scheduling overhead
2. **Ragged Paging** — clusters heads by retention demand with independent, vectorized page tables
3. **Ahead-of-Time (AOT) Load Balancing** — offline workload partitioning for uniform SM utilization

---

## Getting Started

Tangram is built on top of [vLLM](https://github.com/vllm-project/vllm), a fast
and easy-to-use library for LLM inference and serving. See the
[vLLM documentation](https://docs.vllm.ai/en/latest/) for the underlying engine
and supported models.

Install from source:

```bash
pip install -e .
```

## Supported Compression

- SnapKV
- KeyDiff
- ExpectedAttention

**🚧 WIP**

- FastKVzip
- TOVA
- PyramidKV

## Accuracy

RULER 8K.

### Non-uniform

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
<tr><td>qwen3-4b</td><td>93.7</td><td>90.8</td><td>82.5</td><td>71.2</td><td>59.8</td><td>93.8</td><td>93.7</td><td>90.3</td><td>73.8</td><td>93.6</td><td>84.3</td><td>48.9</td><td>22.4</td></tr>
<tr><td>llama3.1-8b</td><td>93.1</td><td>91.8</td><td>87.8</td><td>76.6</td><td>60.6</td><td>92.9</td><td>91.9</td><td>84.9</td><td>73.8</td><td>92.2</td><td>84.4</td><td>61.4</td><td>32.7</td></tr>
<tr><td>gemma3-12b</td><td>91.7</td><td>85.2</td><td>72.3</td><td>61.4</td><td>54.5</td><td>91.3</td><td>89.3</td><td>79.2</td><td>67.9</td><td>90.5</td><td>72.4</td><td>40.0</td><td>21.6</td></tr>
<tr><td>gptoss-20b</td><td>83.7</td><td>83.5</td><td>80.3</td><td>69.4</td><td>56.2</td><td>83.5</td><td>83.4</td><td>71.2</td><td>40.4</td><td>83.7</td><td>74.9</td><td>44.0</td><td>20.3</td></tr>
<tr><td>qwen3-30b</td><td>95.3</td><td>90.8</td><td>85.1</td><td>74.7</td><td>66.6</td><td>95.3</td><td>95.2</td><td>93.8</td><td>81.7</td><td>94.6</td><td>85.5</td><td>51.4</td><td>26.5</td></tr>
</tbody>
</table>

### Uniform

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

## Evaluate on RULER

Run one compression method at one ratio on RULER 8K.

```bash
cd benchmarks/tangram

MODEL=meta-llama/Llama-3.1-8B-Instruct \
SCORER=snapkv LEVEL=crosslayer_cluster RATIOS=0.5 LENGTHS=8192 \
bash benchmark_ruler.sh
```

- `SCORER` — `snapkv` | `keydiff` | `expected_attention`
- `RATIOS` — KV retention fraction (`1.0` = FullKV reference)
- `LEVEL` — `crosslayer_cluster` (non-uniform) | `uniform`

---

## Citation

If you use Tangram for your research, please cite our [paper](https://arxiv.org/abs/2606.06302):

```
@misc{kim2026tangramunlockingnonuniformkv,
      title={Tangram: Unlocking Non-Uniform KV Cache for Efficient Multi-turn LLM Serving}, 
      author={Hyungmin Kim and Minsoo Kim and Hongseok Kim and Jungwook Choi},
      year={2026},
      eprint={2606.06302},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.06302}, 
}
```