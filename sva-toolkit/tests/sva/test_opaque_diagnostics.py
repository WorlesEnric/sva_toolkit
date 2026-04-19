"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T07.

This test module verifies the "silent fallback is dead" guarantee
introduced by T07. It uses a deliberately malformed SVA surface that
today (pre-T07) is silently wrapped in an `OpaqueProperty`, and asserts
after T07 that: (1) `sva_toolkit.sva.parser.parse_property_body(text,
recover=True)` still returns an `OpaqueProperty`, (2)
`sva_toolkit.sva.diagnostics.ParserDiagnostics.snapshot()[
'opaque_property']` incremented, (3) a `logging.WARNING` record was
emitted under the `sva_toolkit` logger, and (4)
`parse_property_body(text, recover=False)` raises `SvaSyntaxError`.
Also ensures that a clean parse leaves every counter at zero. Relates
to DAG task T07.
"""

from __future__ import annotations

# Tests belong to T07. Intentionally empty.
