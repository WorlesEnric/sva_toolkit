# SVA Toolkit V3 — Supported Features (scaffold)

SCAFFOLD SUMMARY — replace this document with the real user-facing
feature inventory in task T15.

This file is the canonical list of what the toolkit **does** support
after the gap-remediation pass described in `docs/task_dag_planning.md`.
It is the document users should consult when evaluating whether
`sva-toolkit` fits their workflow, and it is the document that will be
referenced by the top-level `README.md` capabilities matrix.

Each feature row must have:

- A unique ID (`F-01`, `F-02`, …) that never changes once published.
- A feature name and a one-sentence summary.
- The CLI command(s) that expose it.
- The Python API entry point(s) that expose it.
- The relevant source file(s) in `src/sva_toolkit/...`.
- The relevant test(s) in `tests/...` that prove it.
- Whether an optional extra is required (`timing-render`, `llm`,
  `formal`, `all`).

The sections the T15 worker must produce:

1. **CLI command surface.** Per-command: inputs, outputs, exit codes,
   optional dependencies, worked examples (prefer the existing
   `docs/sva-*.md` examples and link to them).
2. **SVA syntax coverage.** Table of every construct that parses and
   round-trips after T06/T07; link to `docs/gaps.md` §2 for the
   pre-remediation baseline.
3. **Describe engine coverage.** Table of every system function and
   operator with a dedicated NL template after T09.
4. **Formal workflow.** `check`, `equivalent`, `relationship` with
   clock/reset flags, backend selection, and exit-code mapping (T08,
   T13).
5. **Timing DSL.** Constructs supported by the grammar parser (T10)
   and the bridging API surface (T11).
6. **Generation.** Deterministic generation via `--seed` (T04),
   coverage computation, Verible validation when available.
7. **Dataset and benchmark.** Atomic-locked cache (T12), LLM retry
   policy knobs (T12), offline-safe defaults.
8. **Runtime guarantees.** Atomic writes (T02), process-group cleanup
   on POSIX timeouts (T03), diagnostic summary at CLI exit (T02+T13).
9. **Compatibility matrix.** Tested Python versions, tested operating
   systems, tested external tool versions.
10. **Extras.** `timing-render`, `llm`, `formal`, `all` — what each
    one unlocks.

Seed feature rows the worker should expand (not exhaustive):

| ID   | Feature                                                             | Task |
| ---- | ------------------------------------------------------------------- | ---- |
| F-01 | Lexer tolerates `//` and `/* */` comments, strings, attributes      | T01  |
| F-02 | Lexer records backtick directives via preprocessor pass              | T01  |
| F-03 | Full SVA keyword coverage (temporal + structural)                   | T06  |
| F-04 | Parser round-trip for every documented construct                     | T07  |
| F-05 | Opaque downgrade surfaces WARNING + counter                          | T07  |
| F-06 | `sva generate --seed` deterministic output                           | T04  |
| F-07 | `sva formal --clock --clock-edge --reset` mandatory flags           | T08  |
| F-08 | EBMC / VCF template sanitization against reserved words              | T05  |
| F-09 | Subprocess orphan reaping on POSIX                                   | T03  |
| F-10 | Atomic file writes for CLI, cache, formal backends                   | T02  |
| F-11 | LLM retry with backoff and `Retry-After` support                     | T12  |
| F-12 | Multiprocess-safe dataset cache with schema version                  | T12  |
| F-13 | Timing DSL grammar parser with precise error messages                | T10  |
| F-14 | `ExtractionReport` surfaces LOSSY / UNSUPPORTED to the CLI           | T11  |
| F-15 | Describe templates for every lexed `$ident` plus uncertainty marker  | T09  |
| F-16 | Stable CLI exit codes (0/1/2/3/4/5/6/7)                              | T13  |
| F-17 | End-of-run `Diagnostics` summary on CLI                              | T02+T13 |

The final document authored in T15 must expand every row above into a
full section with the eight required fields, include worked examples,
and provide a one-screen "quick capability matrix" at the top. Relates
to DAG task T15.
