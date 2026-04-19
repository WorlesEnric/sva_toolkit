"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T04.

This file centralizes randomness for the generator so that `sva generate
--seed N` can reproduce its output byte-for-byte, a direct fix for
`docs/gaps.md` §3.5 and risk R5. It must expose a `GenerationRng` class
wrapping `random.Random` and a `resolve_seed(explicit: int | None) -> int`
helper that either returns the explicit seed or draws one from
`secrets.randbits(32)` and prints it to stderr (so the reproducer can
capture it). `GenerationRng` must expose only the methods actually used
in `generate/synthesizer.py`, `generate/stratified.py`, and
`generate/utils.py` (`random`, `randint`, `choice`, `choices`, `sample`,
`shuffle`, `uniform`, `seed`) so that migrating those modules is a
one-import mechanical rename and a linter can ban the bare `random`
module there. The constructor accepts a seed (`int | None`) — a `None`
seed triggers `resolve_seed`. The module depends only on the standard
library. Relates to DAG task T04; consumed by T13 via
`cli/generate_flags.py`.
"""

from __future__ import annotations

# Implementation belongs to T04. Intentionally empty.
