"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T02.

This file is the single, foundational atomic-write helper used by every
domain of the toolkit. It must expose three public functions:
`atomic_write_text(path, content, *, encoding="utf-8")`,
`atomic_write_json(path, payload, *, indent=2, sort_keys=True)`, and
`atomic_write_jsonl(path, rows)`. Each writes to a sibling temporary file
(`{path}.tmp.{pid}.{nonce}`) using `os.replace` to publish the final name,
so a Ctrl-C mid-write leaves the destination unchanged. On POSIX the
helpers also `fsync` the parent directory when feasible so crash-
consistency holds. The module is imported by `cli/main.py` (via T13),
`data/dataset.py` (via T12), and `formal/backends/ebmc.py` (via T12/T08)
to replace today's non-atomic `Path.write_text` call sites listed in
`docs/gaps.md` §3.8. It does not depend on any other sva_toolkit module;
it must remain a safe zero-cost import. Relates to DAG task T02.
"""

from __future__ import annotations

# Implementation belongs to T02. Intentionally empty.
