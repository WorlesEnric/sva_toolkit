"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T03.

This test module, POSIX-only (skip on Windows via a pytest skip
marker), verifies that `sva_toolkit.runtime.process.run_tool`
terminates the *entire* child process group on timeout, closing
risk R6 from `docs/gaps.md`. The test spawns a shell script that
forks a grandchild (`sh -c 'sleep 60 & sleep 60; wait'`), triggers a
short timeout via `run_tool(..., timeout=1)`, and after the call
returns asserts that neither the child nor the grandchild PIDs
still exist (probe via `os.kill(pid, 0)` → `ProcessLookupError`).
Also exercises: `ToolMissingError` raised for an absent binary;
`make_work_dir` permissions equal `0o700` on POSIX. Relates to DAG
task T03.
"""

from __future__ import annotations

# Tests belong to T03. Intentionally empty.
