#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva parse "$EXAMPLES_DIR/inputs/parse/req_ack.sv" --format json
