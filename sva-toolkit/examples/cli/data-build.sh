#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva data build "$EXAMPLES_DIR/data/dataset_input.json" -o "$OUT_DIR/dataset.jsonl" --workers 1
echo "Wrote $OUT_DIR/dataset.jsonl"
