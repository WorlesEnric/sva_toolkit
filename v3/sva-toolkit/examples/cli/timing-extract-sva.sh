#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva timing extract-sva "$EXAMPLES_DIR/sva/11_emit_sva_bridge.sv" -o "$OUT_DIR/11_emit_sva_bridge.td"
echo "Wrote $OUT_DIR/11_emit_sva_bridge.td"
