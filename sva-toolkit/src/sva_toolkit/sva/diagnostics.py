"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T07.

This file owns the parser-side diagnostics counter that surfaces the silent
`recover=True` opaque downgrades called out in `docs/gaps.md` §2.2. It must
expose a module-level `ParserDiagnostics` object with thread-safe counters
for `opaque_property`, `opaque_sequence`, `opaque_expr`, and
`fallback_recover_used`, a `reset()` method, a `snapshot()` method that
returns a frozen dict, and a `emit_warning(kind, text, span)` helper that
logs through the shared `sva_toolkit.runtime.diagnostics` façade before
bumping the counter. Every call site in `sva_toolkit.sva.parser` that
currently wraps a parse failure into an `OpaqueProperty`/`OpaqueSequence`/
`OpaqueExpr` must route through the helper so the CLI and the dataset
builder can detect the downgrade and react (either exit non-zero or at
least print a WARNING summary). The module depends only on
`sva_toolkit.runtime.diagnostics`. Relates to DAG task T07.
"""

from __future__ import annotations

# Implementation belongs to T07. Intentionally empty.
