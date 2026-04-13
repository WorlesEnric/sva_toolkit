#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva timing bundle-sva \
  "$EXAMPLES_DIR/sva/11_emit_sva_bridge.sv" \
  "$EXAMPLES_DIR/sva/12_extract_sva_bridge.sv" \
  -o "$OUT_DIR/bundled.td"
echo "Wrote $OUT_DIR/bundled.td"
