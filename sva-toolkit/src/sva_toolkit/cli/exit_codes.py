"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T13.

This file owns the stable CLI exit-code table and the mapping from
typed exceptions to those codes, fixing `docs/gaps.md` §3.9 and risk
R17. It must expose an `ExitCode` `IntEnum` with values
`SUCCESS = 0`, `GENERIC_ERROR = 1`, `USAGE_ERROR = 2`,
`TOOL_MISSING = 3`, `PARSE_ERROR = 4`, `TIMEOUT = 5`,
`LOSSY_EXTRACTION = 6`, `BACKEND_UNAVAILABLE = 7`; and a single
`exit_code_for(exc: BaseException) -> ExitCode` dispatcher that
consults the `runtime.errors` taxonomy (T03), `sva.errors.SvaSyntaxError`,
`timing.bridge.status.LossyExtractionError` (T11), and
`formal.errors.BackendUnavailableError` (T08). The dispatcher is
consumed by `cli/main.py::_handle_cli_errors` in T13 to call
`ctx.exit(code)` with the correct value and print a one-line WARNING
summary from `runtime.diagnostics.Diagnostics` if any silent-fallback
category is non-zero. Dependencies: `runtime.errors`, `sva.errors`,
`timing.bridge.status`, `formal` exceptions. Relates to DAG task T13.
"""

from __future__ import annotations

# Implementation belongs to T13. Intentionally empty.
