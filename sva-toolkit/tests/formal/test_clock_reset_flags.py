"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T08.

This test module verifies the T08 change: `sva formal check` /
`equivalent` / `relationship` no longer silently inject `clk`/`posedge`/
`!rst_n` when the property text omits clocking or reset. Required
cases: (1) a property with explicit `@(posedge clk) disable iff (!rst_n)`
parses and produces a normal `FormalProperty`, (2) a property lacking
clocking and invoked without `--clock` raises
`MissingClockingError` → CLI exit 2 (usage error), (3) a property
lacking reset and invoked without `--reset` raises
`MissingResetError`, (4) equivalence check treats `!rst_n` and
`rst_n == 0` as equivalent via the semantic comparator, (5) the CLI
flags plumb through to both EBMC and VCF backend stubs. Uses the
sanitizer from T05 to validate the CLI flag values. Relates to DAG
task T08.
"""

from __future__ import annotations

# Tests belong to T08. Intentionally empty.
