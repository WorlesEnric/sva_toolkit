# SVA Toolkit V3 — Limitations (scaffold)

SCAFFOLD SUMMARY — replace this document with the real user-facing
limitations inventory in task T15.

This file is the canonical, stable list of everything that the toolkit
deliberately does **not** support after the gap-remediation pass
described in `docs/task_dag_planning.md`. It is the document users
should consult before deciding whether `sva-toolkit` fits their
workflow, and it is the document that downstream consumers should cite
when filing feature requests.

Each row must have:

- A unique ID (`L-01`, `L-02`, …) that never changes once published.
- A category (`syntax`, `robustness`, `integration`, `infrastructure`,
  `platform`).
- A one-sentence description.
- The root cause (why the toolkit does not support it — out of scope,
  requires vendor integration, requires architectural rewrite, etc.).
- The user-visible symptom (what happens if they try it anyway).
- A workaround if one exists.
- A link to the canonical discussion in `docs/gaps.md` or to the
  relevant source file.

Seed entries the worker should expand (not exhaustive):

| ID   | Category       | One-line limitation                                                                               |
| ---- | -------------- | ------------------------------------------------------------------------------------------------- |
| L-01 | syntax         | No full RTL module parsing; `sva parse` remains property-centric per `docs/sva-parse.md`.         |
| L-02 | syntax         | No UVM / OVL macro expansion (risk R12).                                                          |
| L-03 | syntax         | Encrypted-IP (`` `protect`` / `` `endprotect``) regions are detected but not decrypted (R14).    |
| L-04 | syntax         | Vendor extensions (VCS `$smashed_*`, Jasper `cover` options, Questa/Xcelium checker extras) (R11).|
| L-05 | integration    | Structured counterexample reconstruction from EBMC raw output is deferred (R13).                  |
| L-06 | integration    | Coverage integrity: `sva generate --coverage` measures generated-property coverage, not design coverage (R15). |
| L-07 | infrastructure | No performance ceiling; very large inputs (≥10 MB) are not streamed (R16).                        |
| L-08 | infrastructure | Audit trail on `CheckResult` carries backend version + timestamp, but not a hash of inputs (R18 partial). |
| L-09 | platform       | Process-group orphan-kill is POSIX-only; Windows uses best-effort `terminate()` → `kill()` (T03).|
| L-10 | integration    | Custom `base_url` and non-OpenAI LLM endpoints are supported in Python API but not yet on CLI.    |

The final document authored in T15 must contain one row per
unfixed gap with the six fields above, and must cross-link to the
specific source location (`src/sva_toolkit/...`:line) that anchors the
limitation. Relates to DAG task T15.
