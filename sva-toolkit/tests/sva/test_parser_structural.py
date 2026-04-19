"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T07.

This test module covers the structural SVA declarations added in T06/T07:
`sequence … endsequence`, `checker … endchecker`, `let`, `bind`,
`clocking … endclocking`, `default clocking`, `restrict property`,
`expect`, and sequence operators `within`, `matched`, `ended` as
first-class. Each case asserts a parse tree shape, round-trips through
the emitter, and confirms that `ParserDiagnostics.opaque_*` counters
do not increment on valid input. Multi-edge clocking samples
(`@(posedge clk or negedge rst_n)`) are exercised here as well. Local
variable declarations inside properties are verified to preserve
declared types (raw string captured by the lexer type tokens). Relates
to DAG task T07.
"""

from __future__ import annotations

# Tests belong to T07. Intentionally empty.
