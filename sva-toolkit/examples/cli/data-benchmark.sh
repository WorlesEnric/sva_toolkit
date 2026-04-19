#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_env OPENAI_API_KEY
require_env SVA_TOOLKIT_MODEL

sva data benchmark \
  "$EXAMPLES_DIR/data/benchmark_input.json" \
  --model "$SVA_TOOLKIT_MODEL" \
  --workers 1 \
  -o "$OUT_DIR/benchmark_results.json"
echo "Wrote $OUT_DIR/benchmark_results.json"
