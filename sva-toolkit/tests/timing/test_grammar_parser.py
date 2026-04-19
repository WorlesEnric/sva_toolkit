"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T10.

This test module verifies the T10 timing DSL grammar parser. Required
cases: (1) every file under `examples/td/` parses to a
`ScenarioDocument` byte-identical to the one produced by the pre-T10
regex parser (use `dataclasses.asdict` for deep equality),
(2) trailing `# comment` on any line is tolerated,
(3) multi-line declarations (signals with wrapped attribute lists)
parse successfully, (4) a malformed input with a dangling parenthesis
produces a clean `SvaTimingSyntaxError` with a `line:column` anchor.
Relates to DAG task T10.
"""

from __future__ import annotations

# Tests belong to T10. Intentionally empty.
