"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T12.

This file provides a small, dependency-free retry decorator used by
`sva_toolkit.runtime.llm.LLMClient.generate` (and any future transient-
failure call site) to fix the "no retries/backoff/rate-limit handling"
gap called out in `docs/gaps.md` §3.6. It must expose a
`RetryPolicy(max_retries: int = 3, backoff_base: float = 1.0,
backoff_cap: float = 30.0, jitter: bool = True,
retry_on: tuple[type[BaseException], ...] = ())` dataclass and a
`retry(policy)` decorator factory that catches the configured exception
types plus HTTP 429/5xx signalled via a custom
`TransientResponseError(message, retry_after: float | None)`. The
decorator honours `Retry-After` hints, caps wait time at
`backoff_cap`, and applies full-jitter exponential backoff. Each
successful-after-retry execution and each exhausted-retry surface is
recorded through `sva_toolkit.runtime.diagnostics` so the CLI summary
can report it. No network, no side effects beyond sleep and logging.
Relates to DAG task T12.
"""

from __future__ import annotations

# Implementation belongs to T12. Intentionally empty.
