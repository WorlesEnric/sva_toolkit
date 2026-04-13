#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva timing render "$EXAMPLES_DIR/td/01_simple_handshake.td" -o "$OUT_DIR/01_simple_handshake.svg"
echo "Wrote $OUT_DIR/01_simple_handshake.svg"
