"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T02.

This test module covers `sva_toolkit.runtime.atomic_io`. Required cases:
(1) `atomic_write_text` writes content to the target atomically — a
second read yields the full content, (2) a simulated failure in
`os.replace` (monkeypatched to raise) leaves the original file
unchanged, (3) concurrent writes from two threads produce exactly one
well-formed final file, (4) `atomic_write_json` and
`atomic_write_jsonl` delegate to the text helper, (5) parent
directories that do not exist produce a clear error (no silent
swallow). Relates to DAG task T02.
"""

from __future__ import annotations

# Tests belong to T02. Intentionally empty.
