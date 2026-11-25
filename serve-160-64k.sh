#!/usr/bin/env bash
# Final accepted config: 160 experts resident, 64k context, INT8 KV, no speculation.
set -euo pipefail
cd "$(dirname "$0")"
exec env AIBOOK_CONTEXT=65536 AIBOOK_MAX_TOKENS=65536 AIBOOK_KV_DTYPE=int8 \
  AIBOOK_MAMBA=12 AIBOOK_FRACTION=0.68 AIBOOK_GUARD_CAP_GB=24.5 AIBOOK_GUARD_MARGIN_GB=0.1 \
  bash experiment_launch.sh resident 160 "${1:-2400}"
