#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_any_command ebmc vcf

sva formal relationship "req |-> ##[1:3] ack" "req |-> ##2 ack" --backend auto --depth 8 --timeout 60
