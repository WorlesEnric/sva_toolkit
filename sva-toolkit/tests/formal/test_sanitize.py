"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T05.

This test module covers `sva_toolkit.formal.sanitize`. Required cases:
(1) `validate_signal` accepts valid identifiers
(`req`, `ack`, `u_req_0`) and rejects reserved words (`module`,
`wire`, `property`), Unicode, leading digits, and hierarchical paths
(`u_dut.req`), (2) `validate_clock` / `validate_reset` accept the
simple boolean shapes documented in the sanitizer (`!rst`,
`rst == 0`), (3) `escape_body` round-trips a string containing
literal `{`/`}` through `string.Template.safe_substitute` without
raising, (4) a fuzz-style parametrized sweep over curated hostile
inputs classifies each as accepted or rejected per the validator
contract. Relates to DAG task T05.
"""

from __future__ import annotations

# Tests belong to T05. Intentionally empty.
