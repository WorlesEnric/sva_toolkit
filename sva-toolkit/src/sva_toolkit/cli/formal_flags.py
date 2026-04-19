"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T08.

This file carries the `--clock`, `--clock-edge`, `--reset` flags for
`sva formal {check,equivalent,relationship}`, introduced by T08 to
replace the silent hard-coded `clk`/`posedge`/`!rst_n` defaults
(`docs/gaps.md` §3.3 / risk R1). It must expose a single
`register(formal_group: click.Group) -> None` function that attaches
the flags to the existing `formal` subgroup and threads their values
into `sva_toolkit.formal.FormalService`. Missing clocking/reset in the
property text plus missing flags must raise
`MissingClockingError`/`MissingResetError` (defined by T08 in
`formal/model.py`) so that T13 can translate them into distinct CLI
exit codes. The module must not import `cli/main.py`; T13 mounts it.
Dependencies: `click`, `sva_toolkit.formal`, `sva_toolkit.formal.sanitize`
(T05). Relates to DAG task T08; composed by T13.
"""

from __future__ import annotations

# Implementation belongs to T08. Intentionally empty.
