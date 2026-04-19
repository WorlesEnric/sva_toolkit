"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T04.

This test module closes risk R5 from `docs/gaps.md`. Required cases:
(1) `SVASynthesizer(..., rng=GenerationRng(seed=42)).generate_module(...)`
called twice produces byte-identical output, (2) running the same with
seed `43` produces different output, (3) the `StratifiedGenerator`
surface is likewise deterministic with a fixed seed, (4) a grep-level
assertion that `generate/` no longer imports `random` at module scope
(`re.search(r"^import random", file_contents, re.MULTILINE)` returns
`None` across the package), (5) `sva generate --seed 42 --count 3`
invoked through the Click test runner produces a deterministic stdout.
Relates to DAG task T04.
"""

from __future__ import annotations

# Tests belong to T04. Intentionally empty.
