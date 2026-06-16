#!/usr/bin/env bash
# M18.0 exit-gate runner: gemma-3-12b-it head-group paging (compression off) must
# be token-identical to base vLLM. Runs the base reference then the head-group
# config in separate processes (a 12B model per process) and diffs token ids.
#
# Invoke directly (do not wrap). GPU is pinned to physical device 3.
set -euo pipefail

REPO=/workspace/vllm-asp
PROBE="$REPO/tools/swa_models/probes/gemma3_head_group_smoke.py"
OUT_DIR="${OUT_DIR:-/tmp/swa_m18_0}"
mkdir -p "$OUT_DIR"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="$REPO"

echo "=== [1/3] base reference (page_group_size disabled) ==="
python3 "$PROBE" --page-group-size 0 --out "$OUT_DIR/base.json"

echo "=== [2/3] head-group paging (page_group_size=2, compression off) ==="
python3 "$PROBE" --page-group-size 2 --out "$OUT_DIR/pg2.json"

echo "=== [3/3] diff token ids ==="
python3 - "$OUT_DIR/base.json" "$OUT_DIR/pg2.json" <<'PY'
import json, sys
base = json.load(open(sys.argv[1]))["generations"]
pg = json.load(open(sys.argv[2]))["generations"]
assert len(base) == len(pg), "generation count mismatch"
all_match = True
for i, (b, g) in enumerate(zip(base, pg)):
    match = b["token_ids"] == g["token_ids"]
    all_match &= match
    status = "MATCH" if match else "MISMATCH"
    print(f"[{status}] prompt {i}: {b['prompt']!r}")
    if not match:
        print(f"   base: {b['token_ids']}")
        print(f"   pg2 : {g['token_ids']}")
        print(f"   base text: {b['text']!r}")
        print(f"   pg2  text: {g['text']!r}")
print("=" * 40)
print("RESULT:", "ALL TOKEN-IDENTICAL ✅" if all_match else "TOKEN MISMATCH ❌")
sys.exit(0 if all_match else 1)
PY
