#!/usr/bin/env bash
# Manual, fresh worker ONLY, after an operator-approved offline window.
set -euo pipefail
: "${AIBOOK_OVERLAY:?new validated overlay directory required}"
: "${AIBOOK_OFFLINE_SMOKE:?set to YES only after taking the baseline offline separately}"
[[ $AIBOOK_OFFLINE_SMOKE == YES && -f $AIBOOK_OVERLAY/AIBOOK_READY ]] || exit 2
source "$HOME/sglang-m1000/env.sh"
export AIBOOK_OVERLAY="$(realpath "$AIBOOK_OVERLAY")"
export AIBOOK_PACK_ROOT="$HOME/sglang-m1000/routed-packs-v1" AIBOOK_SLOTS="${AIBOOK_SLOTS:-8}"
export AIBOOK_WARMUP_SIZES="${AIBOOK_WARMUP_SIZES:-2048,8192}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$HOME/.cache/sglang}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$HOME/.cache/tilelang/sglang_m1000}"
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
# TileLang 3D attention insists on an exact KV-block capacity bucket; the
# capacity depends on the memory left after the resident banks and MTP draft.
# Provide every integer so the bucket lookup is exact for any capacity.
export AIBOOK_KV_BUCKETS="${AIBOOK_KV_BUCKETS:-$(seq -s, 1 4097)}"
export UA_DECODE_KV_BLOCK_BUCKETS="${UA_DECODE_KV_BLOCK_BUCKETS:-$AIBOOK_KV_BUCKETS}"
export UA_PREFILL_KV_BLOCK_BUCKETS="${UA_PREFILL_KV_BLOCK_BUCKETS:-$AIBOOK_KV_BUCKETS}"
if (( MAXTOK > 65536 )); then
  export UA_DECODE_BLOCK_TABLE_COL_BUCKETS="${UA_DECODE_BLOCK_TABLE_COL_BUCKETS:-512,4097,8193}"
fi
export QUANTIZATION=moe_wna16 MAX_TOTAL_TOKENS="$MAXTOK" MAX_PREFILL_TOKENS="${AIBOOK_MAX_PREFILL_TOKENS:-64}" CHUNKED_PREFILL_SIZE="${AIBOOK_CHUNKED_PREFILL_SIZE:-64}"
# Keep actual KV and Mamba pool sizes fixed; admit the larger routed weights.
if [[ $AIBOOK_RESIDENT == 1 ]]; then
  if [[ ${AIBOOK_MTP:-0} == 1 ]]; then fraction=0.70; else fraction=0.65; fi
  fraction="${AIBOOK_FRACTION:-$fraction}"
else
case "$AIBOOK_SLOTS" in 8) fraction=0.55;; 128) fraction=0.74;; 140) fraction=0.74;; 144) fraction=0.74;; 160) fraction=0.80;; *) exit 2;; esac
fi
export QWEN35_EXTRA_SERVER_ARGS="--mem-fraction-static $fraction --context-length $CTX --max-total-tokens $MAXTOK"
ROUTE_ARGS=()
if [[ $AIBOOK_EAGER == 1 ]]; then
    ROUTE_ARGS=(--enable-return-routed-experts)
fi
KV_ARGS=()
if [[ -n ${AIBOOK_KV_DTYPE:-} ]]; then KV_ARGS=(--kv-cache-dtype "$AIBOOK_KV_DTYPE"); fi
HICACHE_ARGS=()
if [[ ${AIBOOK_HICACHE:-0} == 1 ]]; then
    export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="${AIBOOK_HICACHE_DIR:-$HOME/.cache/sglang-hicache}"
    HICACHE_ARGS=(--enable-hierarchical-cache --hicache-storage-backend file
                  --hicache-ratio "${AIBOOK_HICACHE_RATIO:-1.0}" --hicache-write-policy write_through)
fi
MAMBA_ARGS=()
if [[ -n ${AIBOOK_MAMBA:-} ]]; then MAMBA_ARGS=(--max-mamba-cache-size "$AIBOOK_MAMBA"); fi
DFLASH_ARGS=()
if [[ ${AIBOOK_DFLASH:-0} == 1 ]]; then
    DFLASH_ARGS=(--speculative-algorithm DFLASH --speculative-draft-model-path "$AIBOOK_DRAFT"
                 --speculative-draft-model-quantization unquant --speculative-draft-attention-backend fa3
                 --speculative-dflash-block-size 16 --speculative-draft-window-size 4096)
fi
DECODE_ARGS=()
if [[ -n ${AIBOOK_DECODE_LOG_INTERVAL:-} ]]; then
    DECODE_ARGS=(--decode-log-interval "$AIBOOK_DECODE_LOG_INTERVAL")
fi
NGRAM_ARGS=()
if [[ ${AIBOOK_NGRAM:-0} == 1 ]]; then
    NGRAM_ARGS=(--speculative-algorithm NGRAM
                --speculative-num-draft-tokens "${AIBOOK_NGRAM_DRAFT:-12}"
                --speculative-num-steps "${AIBOOK_NGRAM_STEPS:-1}"
                --speculative-eagle-topk 1 --speculative-ngram-max-bfs-breadth 1
                --speculative-ngram-external-corpus-path "$AIBOOK_NGRAM_CORPUS"
                --speculative-ngram-external-corpus-max-tokens "${AIBOOK_NGRAM_MAX_TOKENS:-7000000}"
                --speculative-ngram-external-sam-budget "${AIBOOK_NGRAM_SAM_BUDGET:-8}"
                --speculative-ngram-capacity "${AIBOOK_NGRAM_CAPACITY:-10000000}"
                --speculative-ngram-max-trie-depth "${AIBOOK_NGRAM_DEPTH:-18}")
fi
MTP_ARGS=()
if [[ ${AIBOOK_MTP:-0} == 1 ]]; then
    MTP_ARGS=(--speculative-algorithm EAGLE --speculative-draft-model-path "${AIBOOK_DRAFT:-$MODEL_PATH}"
              --speculative-num-steps "${AIBOOK_MTP_STEPS:-3}" --speculative-eagle-topk 1 --speculative-draft-attention-backend fa3
              --speculative-num-draft-tokens "${AIBOOK_MTP_DRAFT:-4}"
              --max-mamba-cache-size "${AIBOOK_MAMBA:-12}")
fi
OVERLAP_ARGS=()
if [[ ${AIBOOK_OVERLAP:-0} != 1 ]]; then
    OVERLAP_ARGS=(--disable-overlap-schedule)
fi
WARMUP_ARGS=()
case "${AIBOOK_WARMUP:-short}" in
    0|off|none) WARMUP_ARGS=(--skip-server-warmup) ;;
    short) ;;
    prefill) WARMUP_ARGS=(--warmups aibook_prefill) ;;
    *) echo "unknown AIBOOK_WARMUP=${AIBOOK_WARMUP}" >&2; exit 2 ;;
esac
exec ./scripts/ci/musa/start_qwen36_chn_server.sh --disable-shared-experts-fusion \
    "${GRAPH_ARGS[@]}" \
    "${OVERLAP_ARGS[@]}" "${KV_ARGS[@]}" "${HICACHE_ARGS[@]}" "${MAMBA_ARGS[@]}" "${MTP_ARGS[@]}" "${DFLASH_ARGS[@]}" "${NGRAM_ARGS[@]}" "${DECODE_ARGS[@]}" "${WARMUP_ARGS[@]}" "${ROUTE_ARGS[@]}"
