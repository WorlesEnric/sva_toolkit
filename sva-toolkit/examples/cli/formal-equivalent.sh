#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_any_command ebmc vcf

sva formal equivalent "req |-> ##1 ack" "req |-> ##1 ack" --backend auto --depth 8 --timeout 60
