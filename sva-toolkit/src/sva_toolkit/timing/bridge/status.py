"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T11.

This file turns the internal `ExtractionStatus.{EXACT, LOSSY,
UNSUPPORTED}` field of `sva_toolkit.timing.bridge.from_sva` into a
first-class user-visible signal so that `sva timing extract-sva` and
`sva timing bundle-sva` can bubble a warning and exit with code
`LOSSY_EXTRACTION` (see `cli/exit_codes.py`, T13). Required public
surface: an `ExtractionReport` dataclass with fields
`per_property: dict[str, ExtractionStatus]`, `reasons: list[str]`, and
`worst_status() -> ExtractionStatus`; a `LossyExtractionError`
exception raised when the caller opts-in to strict mode; and a
`summarize_report(report) -> str` rendering routine. The `from_sva`
extractor is modified (T11) to return this object alongside the
`ScenarioDocument` and to replace the four broad `except Exception`
handlers at lines 578/1462/1486/1517 with targeted catches that log
through `runtime.diagnostics` and append a typed reason. Dependencies:
standard library plus `sva_toolkit.runtime.diagnostics` (T02). Relates
to DAG task T11.
"""

from __future__ import annotations

# Implementation belongs to T11. Intentionally empty.
