"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T01.

This test module verifies that `sva_toolkit.sva.lexer.tokenize` and the new
`trivia` helpers correctly handle SystemVerilog source realities from
`docs/gaps.md` §2.3: `//` line comments, `/* ... */` block comments
(including nested-looking but not actually nested per spec), horizontal
and vertical whitespace, `\\` line continuations, escaped identifiers
(`\\name-with-dashes `), and attribute instances `(* attr = "val" *)`.
Tests also verify that unterminated comment/string inputs raise a
precise `SvaSyntaxError` with position information, that `tokenize`
over every `examples/sva/*.sv` file with a synthetic `// header` and an
inline `/* block */` inserted succeeds, and that the returned `Trivia`
list preserves source order and spans. Relates to DAG task T01.
"""

from __future__ import annotations

# Tests belong to T01. Intentionally empty.
