# Robust SVA Toolkit Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the repository into a clean, reliable, productizable `sva-toolkit` with a clear architecture, stable APIs, safe defaults, and no research clutter in the distributable package.

**Architecture:** Keep one package, but split it into canonical product domains: parsing, formal checking, generation, natural-language/data workflows, and timing. Standardize all external-tool and LLM access behind shared adapters, then finish current migrations instead of carrying parallel models indefinitely.

**Tech Stack:** Python 3.11+, `click`, `pytest`, `ruff`, `mypy`, `pydantic`, optional extras for external tools and PNG export.

---

## Current Baseline

Observed on March 13, 2026:

- `pytest -q` reports `13 failed, 113 passed, 2 errors`.
- There is API drift between tests and implementation in benchmark and implication checker code.
- There are committed secrets and environment-specific defaults in CLI/example files.
- The repo contains committed generated outputs, datasets, `.DS_Store`, and `__pycache__` artifacts.
- The `timing` subsystem currently carries both a newer symbolic scenario model and a legacy adapter path.
- External tool lookup is inconsistent across modules.

## Target Product Shape

Keep these as first-class product capabilities:

1. `parse`: SVA parsing and structure extraction.
2. `formal`: implication/equivalence checking with pluggable backends.
3. `generate`: type-directed SVA generation and coverage analysis.
4. `describe`: SVAD / CoT / dataset workflows.
5. `timing`: DSL <-> scenario <-> SVG/SVA conversion.

Delete or move out of the product package:

1. Committed secrets, raw benchmark outputs, scratch JSON/SVG/PDF files, `.DS_Store`, and `__pycache__`.
2. Example configs containing real API keys.
3. Research-only payloads that are not required for install or tests.
4. Compatibility layers that exist only to support unfinished migrations after the migration is completed.

## Phase 1: Stop the Bleeding

### Task 1: Remove secrets and unsafe defaults

**Files:**
- Modify: `src/sva_toolkit/benchmark/cli.py`
- Modify: `src/sva_toolkit/dataset_builder/cli.py`
- Modify: `examples/usage_example.py`
- Modify: `examples/llm_configs.json`
- Modify: `examples/llm_configs_sillicon.json`
- Test: `pytest tests/test_benchmark.py tests/test_dataset_builder.py -q`

**Plan:**
- Replace embedded API keys with environment-variable lookup or required CLI flags.
- Replace vendor-specific default model choices with neutral placeholders.
- Document required env vars in `README.md`.
- If keys are real, rotate them outside the repo immediately.

### Task 2: Purge committed junk and tighten ignore rules

**Files:**
- Modify: `.gitignore`
- Delete: root `*.svg`, `*.pdf`, `*.json`, `.DS_Store` scratch files that are not source artifacts
- Delete: committed `__pycache__` trees under `src/`
- Move or delete: nonessential files under `data/`, `examples/`, and `docs/` that are outputs rather than inputs
- Test: `git status --short`

**Plan:**
- Add ignores for `.DS_Store`, `__pycache__/`, local caches, generated diagrams, benchmark outputs, and local config files.
- Keep only reproducible examples and tiny fixtures in-repo.
- Move large datasets and experiment outputs to `artifacts/` outside the package or to release assets.

### Task 3: Remove accidental test collection from source tree

**Files:**
- Move: `src/sva_toolkit/utils/test_llm_configs.py` -> `scripts/test_llm_configs.py` or `tools/test_llm_configs.py`
- Modify: imports or docs that reference it
- Test: `pytest -q`

**Plan:**
- Utility scripts do not belong inside importable package paths with `test_*.py` names.
- Keep the package import graph clean and stop pytest from treating source utilities as test modules.

## Phase 2: Re-establish Stable Contracts

### Task 4: Repair API drift before structural refactors

**Files:**
- Modify: `src/sva_toolkit/benchmark/runner.py`
- Modify: `src/sva_toolkit/implication_checker/checker.py`
- Modify: `src/sva_toolkit/dataset_builder/builder.py`
- Test: `pytest tests/test_benchmark.py tests/test_implication_checker.py tests/test_dataset_builder.py -q`

**Plan:**
- Restore or rename compatibility helpers expected by tests:
  - benchmark `_clean_sva_output`
  - implication `_extract_property_expression`
  - implication `_collect_signals`
- Make dataset and benchmark code degrade cleanly to single-process mode when given mocks or non-serializable clients.
- Decide whether syntax errors remain a distinct result type or collapse to `ERROR`, then align tests and CLI behavior consistently.

### Task 5: Define a real public API surface

**Files:**
- Modify: `src/sva_toolkit/__init__.py`
- Modify: package `__init__.py` files under `ast_parser`, `benchmark`, `dataset_builder`, `timing`, `gen`
- Add: `src/sva_toolkit/api/` if needed for stable facades
- Test: targeted import smoke tests

**Plan:**
- Stop eagerly exporting half the package from the root module.
- Expose a narrow, documented public API and keep the rest internal.
- Avoid import-time side effects and tool discovery during plain imports.

## Phase 3: Consolidate Shared Infrastructure

### Task 6: Centralize external-tool discovery and execution

**Files:**
- Add: `src/sva_toolkit/runtime/tools.py`
- Add: `src/sva_toolkit/runtime/subprocess.py`
- Modify: `src/sva_toolkit/ast_parser/parser.py`
- Modify: `src/sva_toolkit/implication_checker/checker.py`
- Modify: `src/sva_toolkit/vcformal_implication_checker/checker.py`
- Modify: `src/sva_toolkit/gen/generator.py`
- Modify: `src/sva_toolkit/gen/cli.py`
- Modify: `src/sva_toolkit/utils/verible_wrapper.py`
- Test: parser, implication, and generator unit tests

**Plan:**
- Replace ad hoc relative paths and hardcoded absolute paths with one resolver.
- Define capability checks once: Verible, EBMC, VC Formal, optional CairoSVG.
- Normalize subprocess error handling, timeouts, temp directories, and diagnostics.

### Task 7: Centralize LLM configuration and execution

**Files:**
- Add: `src/sva_toolkit/runtime/llm.py`
- Modify: `src/sva_toolkit/utils/llm_client.py`
- Modify: `src/sva_toolkit/dataset_builder/builder.py`
- Modify: `src/sva_toolkit/benchmark/runner.py`
- Modify: `src/sva_toolkit/svad_translator/translator.py` if LLM integration is added later
- Test: dataset and benchmark tests

**Plan:**
- One config model, one client wrapper, one retry policy, one way to load env defaults.
- Separate pure business logic from multiprocessing orchestration.
- Make offline and mocked testing first-class.

## Phase 4: Simplify the Domain Architecture

### Task 8: Merge formal verification under one package

**Files:**
- Add: `src/sva_toolkit/formal/`
- Move/modify: `src/sva_toolkit/implication_checker/`
- Move/modify: `src/sva_toolkit/vcformal_implication_checker/`
- Modify: `pyproject.toml`
- Modify: CLI modules
- Test: `pytest tests/test_implication_checker.py -q`

**Plan:**
- Present EBMC and VC Formal as backends of one formal-checking subsystem.
- Share result models, signal extraction, module generation, and reporting.
- Keep separate backend adapters, not separate top-level products.

### Task 9: Merge description/data workflows around one pipeline

**Files:**
- Modify: `src/sva_toolkit/svad_translator/translator.py`
- Modify: `src/sva_toolkit/cot_builder/builder.py`
- Modify: `src/sva_toolkit/dataset_builder/builder.py`
- Modify: `src/sva_toolkit/benchmark/runner.py`
- Add: `src/sva_toolkit/describe/` or `src/sva_toolkit/data/`
- Test: `pytest tests/test_cot_builder.py tests/test_dataset_builder.py tests/test_benchmark.py -q`

**Plan:**
- Make `SVADTranslator` the canonical deterministic description engine.
- Reserve LLM calls for optional enrichment instead of baking them into the core path.
- Reuse the same prompt/cleaning/output models across dataset building and benchmark runs.

### Task 10: Finish the timing-model migration

**Files:**
- Modify: `src/sva_toolkit/timing/core/scenario.py`
- Modify: `src/sva_toolkit/timing/frontend/parser.py`
- Modify: `src/sva_toolkit/timing/bridge/emit_sva.py`
- Modify: `src/sva_toolkit/timing/render/svg.py`
- Modify: timing tests under `tests/timing/`
- Test: `pytest tests/timing -q`

**Plan:**
- Choose one canonical timing IR.
- Remove `legacy_diagram` bridging once feature parity exists.
- Keep compatibility at the CLI boundary, not inside the domain model.

### Task 11: Shrink and rationalize the generator package

**Files:**
- Modify: `src/sva_toolkit/gen/`
- Move or regenerate: `src/sva_toolkit/gen/signal_pool_expanded.py`
- Add: scripts or data-generation path if the signal pool remains useful
- Test: `pytest tests/test_full_properties.py -q`

**Plan:**
- The 20k-line signal pool is a maintenance smell unless it is a generated asset with provenance.
- Either replace it with curated presets plus a generation script, or move it to packaged data with explicit regeneration steps.
- Keep the generator focused on SVA synthesis, not repository-sized vocabularies.

## Phase 5: Make It a Toolkit Instead of a Lab Notebook

### Task 12: Unify CLI ergonomics

**Files:**
- Add: `src/sva_toolkit/cli/main.py`
- Modify: existing CLI modules
- Modify: `pyproject.toml`
- Test: CLI smoke tests

**Plan:**
- Add one top-level `sva` command with subcommands:
  - `sva parse`
  - `sva formal`
  - `sva generate`
  - `sva describe`
  - `sva timing`
- Keep existing entry points as thin aliases during migration, then deprecate them.

### Task 13: Fix packaging boundaries and optional extras

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: docs under `docs/`
- Test: editable install and wheel build

**Plan:**
- Define extras such as:
  - `dev`
  - `timing-png`
  - `formal-ebmc`
  - `formal-vcformal`
  - `llm`
- Stop shipping private or machine-specific assumptions in package defaults.
- Ensure `pip install sva-toolkit` gives a sane core package.

### Task 14: Add CI and quality gates

**Files:**
- Add: `.github/workflows/ci.yml`
- Add: `pytest.ini` if useful
- Modify: `Makefile`
- Modify: `pyproject.toml`
- Test: local `pytest -q`, `ruff check`, `ruff format --check`, `python -m build`

**Plan:**
- No refactor should land without:
  - unit tests
  - lint
  - package build
- Add fast CI for pure-Python checks and optional jobs for tool-dependent integration tests.

## Deletion Rules

Delete aggressively if a file is any of the following:

1. Generated locally and reproducible from a script.
2. Environment-specific or machine-specific.
3. A scratch output, benchmark result, or one-off analysis artifact.
4. An adapter retained only because a migration was started but not finished.
5. A duplicate interface for the same domain concept.

Do not delete if the file is:

1. A minimal example used in docs or tests.
2. A fixture that makes tests deterministic.
3. A packaged template or reference asset needed at runtime.

IMPORTANT NOTE: Since it's a clean reconstruction in the new project root directory, you do not actually need to delete anything, by deleting means that you don't need to migrate that part from the old module to the current project.

## Recommended Execution Order

1. Security and hygiene: secrets, junk files, ignore rules.
2. Test-baseline repair: fix the 13 failures and 2 errors without changing behavior.
3. Shared runtime adapters: tools and LLM config.
4. Domain consolidation: formal, describe/data, timing migration.
5. CLI unification and packaging cleanup.
6. CI, docs rewrite, release cut.

## Non-Negotiable Principles

1. No hardcoded absolute paths.
2. No secrets in source, docs, or examples.
3. No research outputs in the installable package.
4. One canonical model per domain.
5. Optional integrations must fail clearly, not at import time.
6. Public APIs must be explicit and test-backed.
7. Every cleanup step must either reduce surface area or improve reliability.
