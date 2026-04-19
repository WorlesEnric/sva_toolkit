"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T14.

This integration test module is POSIX-only and closes risk R6 at the
CLI level. It replaces `ebmc` on PATH with a shell script that forks a
grandchild sleep, invokes `sva formal check ... --timeout 1`, and
after the call asserts that both the stubbed `ebmc` child PID and the
grandchild PID are reaped (probed with `os.kill(pid, 0)` →
`ProcessLookupError`). The test is skipped on Windows via
`pytest.mark.skipif(sys.platform.startswith('win'), ...)` and that
limitation is documented in `docs/LIMITATIONS.md` (T15). Relates to
DAG task T14.
"""

from __future__ import annotations

# Tests belong to T14. Intentionally empty.
