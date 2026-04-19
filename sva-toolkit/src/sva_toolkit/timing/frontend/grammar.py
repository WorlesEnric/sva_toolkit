"""
SCAFFOLD SUMMARY — replace this paragraph with the real implementation in task T10.

This file is the grammar-based replacement for the regex-per-line
timing DSL parser (`docs/gaps.md` §3.2). It must define: a compact
`TdToken` dataclass; a `TdLexer` that produces tokens from the timing
DSL source and tolerates trailing `# …` comments and multi-line
declarations; and a `TdParser` that consumes the token stream into the
same `ScenarioDocument` shape already produced by
`timing/frontend/parser.py::parse_diagram`. Error reporting uses precise
`line:column` positions. The module must preserve the public
`parse_diagram(source: str) -> ScenarioDocument` entry point (imported
and re-exported by `timing/frontend/parser.py`) so that existing
callers in `timing/bridge/*`, `timing/projection/*`, and
`timing/render/*` need no edits. Dependencies:
`sva_toolkit.timing.core.scenario`, `sva_toolkit.timing.errors`.
Relates to DAG task T10.
"""

from __future__ import annotations

# Implementation belongs to T10. Intentionally empty.
