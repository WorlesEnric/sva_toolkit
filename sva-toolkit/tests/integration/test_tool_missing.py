"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T14.

This integration test module confirms that the CLI reports a clean,
typed failure when an optional external tool is not installed. Each
case monkeypatches PATH so that the target binary is absent and then
runs the CLI: (1) `sva formal check ...` with no `ebmc` nor `vcf` →
exit 3, stderr contains the tool name and an install hint,
(2) `sva generate --validate` without `verible-verilog-syntax` → the
command succeeds but prints a WARNING that validation was skipped,
(3) `sva timing render --format png` without `cairosvg` → exit 3 with
the install hint. Closes risk R-tool-missing reporting. Relates to DAG
task T14.
"""

from __future__ import annotations

# Tests belong to T14. Intentionally empty.
