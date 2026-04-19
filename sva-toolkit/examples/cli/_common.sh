#!/usr/bin/env bash
set -euo pipefail

EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$EXAMPLES_DIR/out"

mkdir -p "$OUT_DIR"

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command: $name" >&2
    exit 1
  fi
}

require_any_command() {
  local name
  for name in "$@"; do
    if command -v "$name" >/dev/null 2>&1; then
      return 0
    fi
  done
  echo "This example requires one of: $*" >&2
  exit 1
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Set $name before running this example." >&2
    exit 1
  fi
}
