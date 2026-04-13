#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

sva generate --count 3 --coverage
