"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T03.

This file defines the typed runtime-level exceptions that callers of
`sva_toolkit.runtime.process.run_tool` can catch in order to distinguish
failure modes — a direct fix for `docs/gaps.md` §3.7 and R17. It must
expose: `ToolMissingError(FileNotFoundError)` for the "binary not on
PATH" case (carries `cmd: Sequence[str]` and an optional hint about
installation), `ToolTimeoutError(RuntimeError)` for timeouts after
process-group kill, `ToolCrashError(RuntimeError)` for nonzero exits,
and `BackendUnavailableError(RuntimeError)` reserved for the formal
service when every configured backend is missing. `cli/exit_codes.py`
(T13) dispatches on these types to assign stable exit codes. The module
is a pure taxonomy: no runtime behaviour, no logging, no subprocess
calls, no imports beyond the standard library. Relates to DAG task T03;
T13 consumes the taxonomy.
"""

from __future__ import annotations

# Implementation belongs to T03. Intentionally empty.
