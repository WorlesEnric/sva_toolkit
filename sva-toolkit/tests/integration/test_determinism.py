"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T14.

This integration test module closes risk R5 at the CLI level. It
invokes `sva generate --seed 42 --count 5` twice through
`click.testing.CliRunner`, asserts byte-identical stdout, and then
invokes with `--seed 43` and asserts non-identical stdout. It also
runs `sva data build` in offline mode twice with a deterministic input
JSON and a fixed seed propagated through the generator path, and
verifies identical JSONL output. Finally it asserts that without
`--seed`, the CLI prints the chosen seed on stderr (so users can
reproduce). Relates to DAG task T14.
"""

from __future__ import annotations

# Tests belong to T14. Intentionally empty.
