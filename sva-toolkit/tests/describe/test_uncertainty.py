"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T09.

This test module verifies T09's uncertainty surfacing in the describe
engine. Required cases: (1) a property containing a deliberately
malformed inner sequence (parsed with `recover=True` and yielding an
`OpaqueSequence`) produces a SVAD text containing the `[unverified]`
marker, (2) the CoT builder adds an explicit low-confidence paragraph
when any opaque node is in the tree, (3) clean properties do not
contain the marker, (4) every `$ident` surfaced by the lexer over the
entire `examples/` tree has a dedicated template (exhaustiveness
check). Relates to DAG task T09.
"""

from __future__ import annotations

# Tests belong to T09. Intentionally empty.
