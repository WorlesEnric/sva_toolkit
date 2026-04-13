#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva timing emit-sva "$EXAMPLES_DIR/td/11_emit_sva_bridge.td" -o "$OUT_DIR/11_emit_sva_bridge.sv"
echo "Wrote $OUT_DIR/11_emit_sva_bridge.sv"
