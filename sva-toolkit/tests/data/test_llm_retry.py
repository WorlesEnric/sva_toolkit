"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T12.

This test module covers the retry behaviour added by T12 to
`sva_toolkit.runtime.llm.LLMClient.generate` via the shared
`runtime.retry.retry` decorator (`docs/gaps.md` §3.6 / risk R7).
Required cases: (1) a mock client that returns HTTP 429 twice and then
200 results in a single successful `generate()` call with
`retry_count == 2` and a respected `Retry-After` header,
(2) persistent HTTP 500 triggers the dataset's translator fallback
path and records `translator_fallback` in the diagnostics collector,
(3) the retry decorator never raises `TransientResponseError` to the
caller — it either succeeds or raises `RetryExhaustedError`,
(4) the `LLMConfig` retry-tuning knobs (`max_retries`, `backoff_base`)
are respected. Relates to DAG task T12.
"""

from __future__ import annotations

# Tests belong to T12. Intentionally empty.
