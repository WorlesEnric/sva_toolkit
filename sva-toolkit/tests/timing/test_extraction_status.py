"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T11.

This test module verifies that T11's `ExtractionReport` correctly
surfaces `LOSSY` and `UNSUPPORTED` cases through
`sva_toolkit.timing.bridge.from_sva.extract_sva_scenario`. Required
cases: (1) a clean input produces `worst_status() == EXACT` and an
empty `reasons` list, (2) a property using an operator that the
extractor cannot faithfully render yields
`worst_status() == LOSSY` or `UNSUPPORTED` with at least one typed
reason entry, (3) broad `except Exception:` is no longer used — the
test patches a targeted internal helper to raise a known exception
and verifies the extractor records the exception type by name,
(4) `summarize_report` renders a deterministic, line-anchored summary.
Relates to DAG task T11.
"""

from __future__ import annotations

# Tests belong to T11. Intentionally empty.
