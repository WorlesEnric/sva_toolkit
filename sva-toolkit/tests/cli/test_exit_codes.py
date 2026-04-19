"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T13.

This test module verifies that the CLI produces stable, documented exit
codes (`cli/exit_codes.ExitCode`) for each failure taxonomy introduced
by T03, T07, T08, T11, T12 (`docs/gaps.md` §3.9 / risk R17). Required
cases, driven through `click.testing.CliRunner`:
(1) `sva formal check ...` without an `ebmc`/`vcf` on PATH → exit 3,
(2) `sva parse "not-valid-sva"` (after T07) → exit 4,
(3) `sva data build` with a mocked LLM client that times out → exit 5,
(4) `sva timing extract-sva` on an input with an UNSUPPORTED operator
    → exit 6,
(5) `--verbose` emits a full traceback and the diagnostics summary on
    stderr, (6) a clean run emits exit 0 and no diagnostics summary.
Relates to DAG task T13.
"""

from __future__ import annotations

# Tests belong to T13. Intentionally empty.
