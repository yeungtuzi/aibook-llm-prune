#!/usr/bin/env bash
# Manual, fresh worker ONLY, after an operator-approved offline window.
set -euo pipefail
: "${AIBOOK_OVERLAY:?new validated overlay directory required}"
: "${AIBOOK_OFFLINE_SMOKE:?set to YES only after taking the baseline offline separately}"
[[ $AIBOOK_OFFLINE_SMOKE == YES && -f $AIBOOK_OVERLAY/AIBOOK_READY ]] || exit 2
source "$HOME/sglang-m1000/env.sh"
export AIBOOK_OVERLAY="$(realpath "$AIBOOK_OVERLAY")"
export AIBOOK_PACK_ROOT="$HOME/sglang-m1000/routed-packs-v1" AIBOOK_SLOTS="${AIBOOK_SLOTS:-8}"
export AIBOOK_RESIDENT="${AIBOOK_RESIDENT:-0}"
export AIBOOK_PRUNE="${AIBOOK_PRUNE:-$AIBOOK_OVERLAY/prune-$AIBOOK_SLOTS.json}"
export MODEL_PATH="$HOME/models/gptq-Qwen3.6-35B-A3B-4bit-channel-lmhead"
export PYTHONPATH="$AIBOOK_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$HOME/sglang-m1000/slot-smoke-work"; mkdir -p "$TMPDIR"
cd "$HOME/sglang-m1000/pkg"
python3 -B - <<'PY'
import importlib.util, os
from pathlib import Path
assert Path(importlib.util.find_spec('sglang').origin).resolve() == Path(os.environ['AIBOOK_OVERLAY'])/'sglang/__init__.py'
PY
export MATE_TILELANG_FORBID_COMPILE=0  # eager attention uses keys outside the vendor graph cache
CTX="${AIBOOK_CONTEXT:-16384}"
MAXTOK="${AIBOOK_MAX_TOKENS:-16384}"
export HOST=127.0.0.1 PORT=30141 MUSA_SERVICE_EAGER="${AIBOOK_EAGER:-1}" QWEN36_MAX_TOTAL_TOKENS="$MAXTOK"
export AIBOOK_EAGER="${AIBOOK_EAGER:-1}"
GRAPH_ARGS=()
if [[ $AIBOOK_EAGER == 1 ]]; then
    GRAPH_ARGS=(--cuda-graph-backend-decode=disabled --cuda-graph-backend-prefill=disabled)
elif [[ $AIBOOK_RESIDENT == 1 ]]; then
    # Full decode graph; skip the 12-shape prefill capture on this memory-tight run.
    GRAPH_ARGS=(--cuda-graph-backend-prefill=disabled)
elif [[ ${AIBOOK_GRAPH_PROBE:-0} == 1 ]]; then
    # Isolate decode capture; a 1024-token prefill capture would serialize the
    # slot path for a very long time and is not the question here. Breakable
    # decode splits graph segments at the explicitly marked routed module.
    GRAPH_ARGS=(--cuda-graph-backend-prefill=disabled --cuda-graph-backend-decode=breakable)
fi
export SGLANG_MUSA_ENABLE_QWEN_SHARED_EXPERT_FUSION=0 SGLANG_MUSA_FUSED_MOE_EPILOGUE=0
# Text-only deployment: no vision tower and no multimodal feature pools.
export SGLANG_VLM_CACHE_SIZE_MB=0 SGLANG_MM_FEATURE_CACHE_MB=0
if (( MAXTOK > 65536 )); then
  # Decode 3D attention requires the exact KV-block capacity bucket.
  export UA_DECODE_KV_BLOCK_BUCKETS="${UA_DECODE_KV_BLOCK_BUCKETS:-1578}"
  export UA_DECODE_BLOCK_TABLE_COL_BUCKETS="${UA_DECODE_BLOCK_TABLE_COL_BUCKETS:-512,4097,8193}"
fi
export QUANTIZATION=moe_wna16 MAX_TOTAL_TOKENS="$MAXTOK" MAX_PREFILL_TOKENS=32 CHUNKED_PREFILL_SIZE=32
# Keep actual KV and Mamba pool sizes fixed; admit the larger routed weights.
if [[ $AIBOOK_RESIDENT == 1 ]]; then fraction=0.65; else
case "$AIBOOK_SLOTS" in 8) fraction=0.55;; 128) fraction=0.74;; 140) fraction=0.74;; 144) fraction=0.74;; 160) fraction=0.80;; *) exit 2;; esac
fi
export QWEN35_EXTRA_SERVER_ARGS="--mem-fraction-static $fraction --context-length $CTX --max-total-tokens $MAXTOK"
ROUTE_ARGS=()
if [[ $AIBOOK_EAGER == 1 ]]; then
    ROUTE_ARGS=(--enable-return-routed-experts)
fi
MTP_ARGS=()
if [[ ${AIBOOK_MTP:-0} == 1 ]]; then
    MTP_ARGS=(--speculative-algorithm EAGLE --speculative-draft-model-path "$MODEL_PATH"
              --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi
OVERLAP_ARGS=()
if [[ ${AIBOOK_OVERLAP:-0} != 1 ]]; then
    OVERLAP_ARGS=(--disable-overlap-schedule)
fi
exec ./scripts/ci/musa/start_qwen36_chn_server.sh --disable-shared-experts-fusion \
    "${GRAPH_ARGS[@]}" \
    "${OVERLAP_ARGS[@]}" "${MTP_ARGS[@]}" --skip-server-warmup "${ROUTE_ARGS[@]}"
