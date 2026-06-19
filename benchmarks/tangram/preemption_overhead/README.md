# KV-cache admission control — preemption benchmark

When the KV cache is under pressure, vLLM's default scheduler can **over-admit**
requests: under chunked prefill it admits a request as soon as its *first chunk*
fits, not its whole input. Those requests then grow past the cache, get
**preempted** (their KV is dropped and recomputed from scratch), and throughput
collapses into preemption thrashing.

The `scheduler_reserve_full_isl` engine option admits a request only if its
*entire* input fits, preventing the over-admission. This benchmark measures the
difference: it runs the same KV-pressured workload with the option **off** then
**on**, and reports preemptions and throughput for each.

## Run

```bash
# Full OFF-vs-ON comparison (pin a free GPU as usual):
CUDA_VISIBLE_DEVICES=0 python benchmark.py
```

Example output:

```
══════════════════════════════════════════════════════════════════
 KV-cache admission control — preemption benchmark
══════════════════════════════════════════════════════════════════
 model      Qwen3-4B-Instruct-2507
 workload   24 requests, ~24646 prompt tokens each
 KV cache   gpu_memory_utilization=0.15 (intentionally tight)
──────────────────────────────────────────────────────────────────
 admission control          preemptions   end-to-end     throughput
 OFF (greedy, default)              440      186.4 s      16.5 tok/s
 ON  (full-input reserve)             0       27.3 s     112.7 tok/s
──────────────────────────────────────────────────────────────────
 → admission control removed 440 preemptions and ran 6.84× faster
══════════════════════════════════════════════════════════════════
```

## Options

```bash
python benchmark.py --gpu-memory-utilization 0.12 --num 32   # tighter cache, more pressure
python benchmark.py --reserve off                            # run one config only
python benchmark.py --help                                   # all options
```

| option | default | meaning |
|---|---|---|
| `--model` | `Qwen/Qwen3-4B-Instruct-2507` | any HF model id / local path |
| `--dataset` | `scbench_kv_short` | long-context scbench dataset |
| `--num` | 24 | number of requests |
| `--gpu-memory-utilization` | 0.15 | **lower = tighter cache = more preemption** |
| `--max-tokens` | 128 | output tokens per request |
| `--watermark` | 0.0 | fraction of KV kept free as decode headroom |

## Notes

- **Preemptions are read from `vllm:num_preemptions`** via `LLM.get_metrics()`,
  not from log lines (the periodic stderr `Preemptions:` line is per-interval and
  unreliable). Let runs finish so the metric is captured.
- The defaults intentionally size the cache to hold only ~3–4 of the requests,
  so the over-admission is visible. Give the cache plenty of room (e.g.
  `--gpu-memory-utilization 0.9`) and both configs behave the same — there is
  nothing to preempt.
