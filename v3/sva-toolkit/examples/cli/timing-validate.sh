#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva timing validate "$EXAMPLES_DIR/td/01_simple_handshake.td"
