"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T12.

This test module covers the cache-race and atomicity fixes in T12
(`docs/gaps.md` §3.6 / risk R8). Required cases: (1) concurrent
`DatasetBuilder.build_dataset` runs with `use_multiprocessing=True` and
a shared `cache_dir` over the same input produce zero corrupted JSON
files (stress: 4 workers, 64 items, each file parseable at the end),
(2) a legacy cache file missing the `__cache_schema` sentinel is
ignored and re-generated rather than silently consumed, (3) the
`_write_cached_result` call uses the T02 `atomic_write_json` helper
(patch-based verification). Relates to DAG task T12.
"""

from __future__ import annotations

# Tests belong to T12. Intentionally empty.
