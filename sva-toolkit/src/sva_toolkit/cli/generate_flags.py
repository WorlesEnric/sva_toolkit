"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T04.

This file is a small CLI flag module carved out so that T04 can ship the
`sva generate --seed` flag without racing the other workers on
`cli/main.py`. T13 will import and mount it. The module must expose a
single `register(generate_group: click.Group) -> None` function that
adds a `--seed INTEGER` option to the existing `sva generate` command
and wires it through to `GenerationRng`. The file must not import
`cli/main.py` (that direction of dependency flows from T13 only). It
depends on `sva_toolkit.generate.rng.GenerationRng` (T04) and on
`sva_toolkit.runtime.diagnostics.configure_cli_logging` (T02) for the
stderr "seed used" notice. Relates to DAG task T04; composed by T13.
"""

from __future__ import annotations

# Implementation belongs to T04. Intentionally empty.
