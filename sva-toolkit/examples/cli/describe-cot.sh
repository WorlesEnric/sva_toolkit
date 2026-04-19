#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva describe cot "$EXAMPLES_DIR/inputs/parse/req_ack.sv"
