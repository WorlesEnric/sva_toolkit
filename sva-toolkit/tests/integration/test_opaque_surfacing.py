"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T14.

This integration test module verifies that silent fallbacks are no
longer silent end-to-end through the CLI (closes risk R2). It runs
`sva parse`, `sva describe svad`, `sva describe cot`, and
`sva data build` against a property that deliberately exercises an
unsupported SVA construct and asserts: (1) each command exits non-zero
with `ExitCode.PARSE_ERROR` or surfaces an `[unverified]` marker on
stdout depending on the command's contract, (2) stderr contains the
diagnostics summary printed by `Diagnostics.render_summary()`,
(3) the dataset output row's `metadata.svad_source` tags
`translator_fallback` when applicable. Relates to DAG task T14.
"""

from __future__ import annotations

# Tests belong to T14. Intentionally empty.
