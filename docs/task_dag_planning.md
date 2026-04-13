IGNORE THIS DOCUMENT IF YOU ARE A CODING LLM AGENT. THIS FILE IS FOR HUMAN REVIEW AND TASK DESIGN ONLY.

# SVA Toolkit V3 Consolidation — Task DAG Planning Document

---

# 1. Project Understanding

## Purpose

The SVA Toolkit is a comprehensive ecosystem for SystemVerilog Assertion (SVA) generation, validation, formal verification, and AI-assisted hardware verification workflows. It targets hardware verification engineers and AI/ML researchers working on LLM-based SVA generation.

## Current State

The codebase is split across two versions:

**V1** (`v1/`) — Mature, feature-complete, Python 3.9+. Contains 10 CLI tools as separate entry points:
- `sva-ast` — Verible-based SVA parsing and structure extraction
- `sva-gen` — Type-directed SVA synthesis with NL IR and coverage analysis
- `sva-implication` — EBMC-based formal implication checking
- `sva-vcformal-implication` — VC Formal-based implication checking
- `sva-cot` — Chain-of-Thought reasoning generation from SVA AST
- `sva-dataset` — LLM-annotated training dataset construction (multiprocessing)
- `sva-benchmark` — LLM evaluation on SVA generation tasks (multiprocessing)
- `svad-translator` — SVA-to-natural-language SVAD translation
- `sva-timing` / `sva-diagram` — Timing diagram DSL ↔ SVG/SVA conversion

Issues: API drift (13 test failures), committed secrets, hardcoded paths, inconsistent tool discovery, junk files, no CI.

**V2** (`v2/sva-tools/`) — Clean rebuild, Python 3.11+, alpha (v2.0.0a1). Covers a subset of V1:
- Custom SVA lexer/parser/AST (replaces Verible dependency for parsing)
- Unified formal verification service with pluggable backends (EBMC, VCFormal)
- Enhanced timing diagram system with graph-based core and WaveDrom integration
- Runtime infrastructure (tool registry, process management, config)
- Single unified CLI entry point (`sva`) with `formal` and `timing` subcommand groups
- GitHub Actions CI, ruff linting, organized test suite (24 test files)

Missing from V2: SVA generator, CoT builder, dataset builder, benchmark runner, SVAD translator, LLM client infrastructure, NL IR.

## Target Architecture (V3)

A single consolidated CLI package at `v3/sva-toolkit/` with these product domains:

| Domain | CLI Group | Source |
|--------|-----------|--------|
| `sva` (parsing) | `sva parse` | V2's `sva/` module (custom parser) |
| `formal` | `sva formal` | V2's `formal/` module |
| `timing` | `sva timing` | V2's `timing/` module |
| `generate` | `sva generate` | Port from V1's `gen/` |
| `describe` | `sva describe` | Port from V1's `svad_translator/`, `cot_builder/` |
| `data` | `sva data` | Port from V1's `dataset_builder/`, `benchmark/` |
| `runtime` | (internal) | V2's `runtime/` + V1's LLM client |

## Major Subsystems

1. **SVA Parser** — Lexer, parser, AST, emitter, visitors, analysis, transforms
2. **Formal Verification** — Backend abstraction, EBMC adapter, VCFormal adapter, property models
3. **Timing Diagrams** — DSL frontend, core model, SVA bridge, rendering (SVG/PNG/WaveDrom)
4. **SVA Generator** — Type-directed synthesis, coverage analysis, NL IR, signal pools
5. **Description Engine** — SVAD translator, CoT builder (template-based reasoning)
6. **Data Workflows** — Dataset builder (LLM + CoT annotation), benchmark runner (LLM evaluation)
7. **Runtime** — Tool registry, process execution, LLM client, configuration

## Constraints

- Python 3.11+ (following V2's baseline)
- Minimal core dependencies: `click`, `pydantic`
- Optional extras for heavy dependencies: `openai` (LLM), `wavedrom`, `cairosvg`, `jinja2`, `rich`
- External tools (EBMC, VCFormal, Verible) discovered at runtime, not import time
- No secrets, no hardcoded paths, no research artifacts in the distributable package

---

# 2. Assumptions and Observed Discrepancies

## Assumptions

1. **V3 is a new directory** (`v3/sva-toolkit/`), not an in-place modification of V1 or V2.
2. **V2's architecture is the foundation** — its module structure, coding style (ruff, Python 3.11+), and patterns are the target.
3. **V2's custom SVA parser replaces Verible** for parsing tasks. Verible remains optional for syntax validation in the generator.
4. **V1's generator module** needs adaptation to work with V2's AST types rather than V1's Verible-based parser.
5. **LLM-dependent modules** (dataset builder, benchmark) should work offline with mocks and degrade gracefully.
6. **The SVAD translator** (966 lines of template-based translation) is ported as-is with interface cleanup.
7. **V1's `ast_parser/` module** is superseded by V2's `sva/` module. The Verible wrapper becomes an optional validation tool in `runtime/`.

## Observed Discrepancies

| Area | V1 | V2 | Resolution |
|------|----|----|------------|
| SVA Parsing | Verible-based (`ast_parser/`) | Custom lexer/parser (`sva/`) | Use V2's parser; keep Verible as optional validator |
| Implication Checker | Two separate packages | Unified `FormalService` | Use V2's unified approach |
| Timing model | Legacy + symbolic scenario | Graph-based core + WaveDrom | Use V2's enhanced model |
| LLM access | `utils/llm_client.py` (OpenAI) | Not present | Port from V1, add to `runtime/llm.py` |
| Generator | Full module with NL IR | Not present | Port from V1, adapt to V2 AST |
| CoT Builder | Depends on V1's `ast_parser` | Not present | Port, rewire to V2's `sva/` parser |
| Test baseline | 13 failures, 2 errors | Clean (all passing) | V3 starts clean |
| Python version | 3.9+ | 3.11+ | 3.11+ |

---

# 3. Task Decomposition Strategy

## Boundary Selection Rationale

Tasks are decomposed along **module boundaries** that match the target package structure. Each task creates or ports one self-contained subpackage under `src/sva_toolkit/`. This ensures:

- Each worker operates in a distinct directory subtree
- Merge conflicts are minimized (different workers touch different packages)
- Each task delivers a testable, importable module

## Parallelization Strategy

- **Wave 1** (T1): Project scaffold — must complete first as all other tasks depend on the directory structure and `pyproject.toml`.
- **Wave 2** (T2, T3): Foundation modules (`runtime/`, `sva/`) — can run in parallel after T1. These are dependencies for all domain modules.
- **Wave 3** (T4, T5, T6, T7): Domain modules — fully parallel after T2+T3. Each creates its own subpackage.
- **Wave 4** (T8): Data workflows — depends on T4 (formal) and T7 (describe) since benchmark uses implication checking and dataset builder uses CoT/SVAD.
- **Wave 5** (T9): Unified CLI — depends on all domain modules being complete.
- **Wave 6** (T10, T11): Documentation and integration testing — parallel, after CLI is wired.

## Merge Conflict Minimization

**Shared hot files:**
- `pyproject.toml` — T1 creates the complete version; T9 adds CLI entry points. No other task modifies it.
- `src/sva_toolkit/__init__.py` — T1 creates a minimal version. T9 updates exports. No other task modifies it.
- `src/sva_toolkit/cli/main.py` — Exclusively owned by T9.

**Coordination rule:** Domain tasks (T4–T8) MUST NOT modify `pyproject.toml`, `__init__.py` at the package root, or `cli/main.py`. They create their own subpackage with its own `__init__.py` and tests.

## Testing Partition

Each domain task includes unit tests for its module. T11 adds cross-module integration tests and CI validation.

---

# 4. Task DAG Summary

```
T1 (Scaffold)
├── T2 (Runtime) ──────────┐
│   ├── T4 (Formal) ───────┤
│   ├── T5 (Timing) ───────┤
│   ├── T6 (Generate) ─────┼── T9 (CLI) ── T10 (Docs)
│   └── T7 (Describe) ─┐   │              T11 (Integration Tests)
│                       └── T8 (Data) ─┘
└── T3 (SVA Parser) ───────┘
```

| Task ID | Name | Depends On | Parallel With |
|---------|------|------------|---------------|
| T1 | Project Scaffold | — | — |
| T2 | Runtime Infrastructure | T1 | T3 |
| T3 | SVA Parser Module | T1 | T2 |
| T4 | Formal Verification | T2, T3 | T5, T6, T7 |
| T5 | Timing Diagrams | T2, T3 | T4, T6, T7 |
| T6 | SVA Generator | T2, T3 | T4, T5, T7 |
| T7 | Description Engine | T2, T3 | T4, T5, T6 |
| T8 | Data Workflows | T2, T4, T7 | — |
| T9 | Unified CLI | T4, T5, T6, T7, T8 | — |
| T10 | Documentation & Examples | T9 | T11 |
| T11 | Integration Tests & CI | T9 | T10 |

**Parallel execution groups:**
- Group A: T1
- Group B: T2, T3
- Group C: T4, T5, T6, T7
- Group D: T8
- Group E: T9
- Group F: T10, T11

---

# 5. Detailed Task Specifications

## T1: Project Scaffold

- **Task ID:** T1
- **Task Name:** Project Scaffold
- **Objective:** Create the V3 project directory structure at `v3/sva-toolkit/` with `pyproject.toml`, directory layout, placeholder `__init__.py` files, CI configuration, linting config, and Makefile.
- **Why this task exists:** All other tasks need a target directory structure and package configuration to work within. This is the foundation that enables parallel development.
- **Inputs / prerequisites:** None. Reference V2's `pyproject.toml` and CI config as templates.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/pyproject.toml`
  - `v3/sva-toolkit/Makefile`
  - `v3/sva-toolkit/.github/workflows/ci.yml`
  - `v3/sva-toolkit/src/sva_toolkit/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/cli/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/formal/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/timing/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/generate/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/describe/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/data/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/runtime/__init__.py`
  - `v3/sva-toolkit/tests/__init__.py` (and subpackage inits)
  - `v3/sva-toolkit/README.md`
  - `v3/sva-toolkit/.gitignore`
- **Files or directories that should be avoided:** V1 and V2 directories (read-only reference).
- **Expected deliverables:**
  - Complete `pyproject.toml` with all dependency groups (core, dev, llm, timing-render, formal-ebmc, formal-vcformal)
  - Directory tree with empty `__init__.py` placeholders for all subpackages
  - GitHub Actions CI workflow (lint, test, build)
  - Makefile with install, test, lint, format, clean targets
  - `.gitignore` with comprehensive Python/IDE/OS patterns
  - Minimal `README.md` with project description
- **Suggested validation:**
  - `pip install -e ".[dev]"` succeeds
  - `ruff check src/` passes
  - `pytest` runs (0 tests collected is OK)
  - `python -m build` produces a wheel
- **Merge conflict risk:** LOW — this task runs alone in Wave 1.
- **Definition of done:** The V3 project can be installed in editable mode, linted, and built. All subpackage directories exist with `__init__.py` files.

---

## T2: Runtime Infrastructure

- **Task ID:** T2
- **Task Name:** Runtime Infrastructure
- **Objective:** Port and consolidate the runtime module: tool registry, process execution, configuration models, and LLM client.
- **Why this task exists:** All domain modules depend on runtime infrastructure for tool discovery (EBMC, VCFormal, Verible), subprocess execution, and LLM access.
- **Inputs / prerequisites:** T1 (scaffold) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/src/sva_toolkit/runtime/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/runtime/config.py`
  - `v3/sva-toolkit/src/sva_toolkit/runtime/tools.py`
  - `v3/sva-toolkit/src/sva_toolkit/runtime/process.py`
  - `v3/sva-toolkit/src/sva_toolkit/runtime/llm.py` (NEW — port from V1)
  - `v3/sva-toolkit/tests/runtime/__init__.py`
  - `v3/sva-toolkit/tests/runtime/test_tools.py`
  - `v3/sva-toolkit/tests/runtime/test_process.py`
  - `v3/sva-toolkit/tests/runtime/test_llm.py`
- **Files to avoid:** `cli/main.py`, other domain subpackages.
- **Expected deliverables:**
  - `ToolRegistry` with `create_default_registry()` (EBMC, VCFormal, Verible)
  - `RunResult` dataclass and `run_tool()` function with timeout support
  - `ToolkitConfig` and `ToolConfig` pydantic/dataclass models
  - `LLMClient` and `LLMConfig` (ported from V1, env-var based, no hardcoded keys)
  - `make_work_dir()` utility
  - Unit tests for all components (LLM tests use mocks)
- **Suggested validation:**
  - `pytest tests/runtime/ -q` — all pass
  - `ruff check src/sva_toolkit/runtime/`
  - No import-time side effects (importing the module doesn't call `shutil.which`)
- **Merge conflict risk:** LOW — isolated to `runtime/` directory.
- **Definition of done:** Runtime module is importable, tested, and provides tool discovery, process execution, and LLM client capabilities.

---

## T3: SVA Parser Module

- **Task ID:** T3
- **Task Name:** SVA Parser Module
- **Objective:** Port V2's custom SVA parser (lexer, parser, AST, emitter, visitors, analysis, transforms, lowerings) to V3.
- **Why this task exists:** The SVA parser is the foundational data model used by formal verification, timing, generator, and description modules.
- **Inputs / prerequisites:** T1 (scaffold) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/src/sva_toolkit/sva/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/lexer.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/parser.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/ast.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/emitter.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/errors.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/analysis.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/transforms.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/visitors.py`
  - `v3/sva-toolkit/src/sva_toolkit/sva/lowerings/` (conditions.py)
  - `v3/sva-toolkit/tests/sva/` (all test files)
- **Files to avoid:** `cli/main.py`, other domain subpackages.
- **Expected deliverables:**
  - Complete SVA parser module ported from V2
  - All V2 SVA tests ported and passing
  - Public API exports in `__init__.py`
- **Suggested validation:**
  - `pytest tests/sva/ -q` — all pass
  - `ruff check src/sva_toolkit/sva/`
  - Roundtrip test: parse → emit → parse produces identical AST
- **Merge conflict risk:** LOW — isolated to `sva/` directory.
- **Definition of done:** SVA parser module is fully functional with all V2 tests passing.

---

## T4: Formal Verification Module

- **Task ID:** T4
- **Task Name:** Formal Verification Module
- **Objective:** Port V2's formal verification module (FormalService, EBMC backend, VCFormal backend, property models, parse utilities) to V3.
- **Why this task exists:** Formal verification is a core product capability used by the CLI and by the benchmark/data workflows.
- **Inputs / prerequisites:** T2 (runtime) and T3 (SVA parser) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/src/sva_toolkit/formal/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/formal/model.py`
  - `v3/sva-toolkit/src/sva_toolkit/formal/parse.py`
  - `v3/sva-toolkit/src/sva_toolkit/formal/normalize.py`
  - `v3/sva-toolkit/src/sva_toolkit/formal/service.py`
  - `v3/sva-toolkit/src/sva_toolkit/formal/backends/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/formal/backends/ebmc.py`
  - `v3/sva-toolkit/src/sva_toolkit/formal/backends/vcformal.py`
  - `v3/sva-toolkit/tests/formal/` (all test files)
- **Files to avoid:** `cli/main.py`, `runtime/`, `sva/`.
- **Expected deliverables:**
  - `FormalService` with pluggable backends
  - `FormalProperty`, `CheckResult`, `ImplicationResult` models
  - EBMC and VCFormal backend adapters using `runtime.tools` and `runtime.process`
  - Property parsing via `sva.parser`
  - Unit tests with mocked backends
- **Suggested validation:**
  - `pytest tests/formal/ -q` — all pass
  - `ruff check src/sva_toolkit/formal/`
- **Merge conflict risk:** LOW — isolated to `formal/` directory.
- **Definition of done:** Formal module is importable, tested, and provides implication/equivalence checking.

---

## T5: Timing Diagrams Module

- **Task ID:** T5
- **Task Name:** Timing Diagrams Module
- **Objective:** Port V2's timing diagram module (frontend, core, bridge, projection, render) to V3.
- **Why this task exists:** Timing diagrams are a core product capability for DSL ↔ SVG/SVA conversion.
- **Inputs / prerequisites:** T2 (runtime) and T3 (SVA parser) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/src/sva_toolkit/timing/` (entire subtree)
  - `v3/sva-toolkit/tests/timing/` (all test files)
- **Files to avoid:** `cli/main.py`, `runtime/`, `sva/`.
- **Expected deliverables:**
  - Complete timing module ported from V2
  - Frontend parser, core model (scenario, conditions, graph)
  - Bridge (emit_sva, from_sva, to_dsl, solver, ebmc_witness)
  - Projection (scenario_view, wavedrom_view)
  - Render (svg, png, wavedrom, waveform)
  - All V2 timing tests ported and passing
- **Suggested validation:**
  - `pytest tests/timing/ -q` — all pass
  - `ruff check src/sva_toolkit/timing/`
- **Merge conflict risk:** LOW — isolated to `timing/` directory.
- **Definition of done:** Timing module is fully functional with all V2 tests passing.

---

## T6: SVA Generator Module

- **Task ID:** T6
- **Task Name:** SVA Generator Module
- **Objective:** Port V1's SVA generator (`gen/`) to V3 as `generate/`, adapting it to V3's architecture and coding standards.
- **Why this task exists:** The SVA generator is a key V1 capability not present in V2. It provides type-directed SVA synthesis, coverage analysis, and NL IR.
- **Inputs / prerequisites:** T2 (runtime) and T3 (SVA parser) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/src/sva_toolkit/generate/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/generate/synthesizer.py` (from V1 `gen/generator.py`)
  - `v3/sva-toolkit/src/sva_toolkit/generate/types.py` (from V1 `gen/types_sva.py`)
  - `v3/sva-toolkit/src/sva_toolkit/generate/coverage.py`
  - `v3/sva-toolkit/src/sva_toolkit/generate/templates.py`
  - `v3/sva-toolkit/src/sva_toolkit/generate/utils.py`
  - `v3/sva-toolkit/src/sva_toolkit/generate/stratified.py`
  - `v3/sva-toolkit/src/sva_toolkit/generate/nl/` (IR, extractor, realizer, symbolic, templates)
  - `v3/sva-toolkit/tests/generate/`
- **Files to avoid:** `cli/main.py`, `runtime/`, `sva/`, other domain packages.
- **Expected deliverables:**
  - `SVASynthesizer` class adapted to V3 patterns
  - Type system (`types.py`) cleaned up
  - Coverage analysis module
  - NL IR subsystem (ir, extractor, realizer, symbolic, templates)
  - Stratified generation support
  - Signal pool either replaced with curated presets + generation script, or moved to package data
  - Verible syntax validation uses `runtime.tools` for discovery
  - Unit tests covering generation, validation, coverage
- **Suggested validation:**
  - `pytest tests/generate/ -q` — all pass
  - `ruff check src/sva_toolkit/generate/`
  - Generated SVA properties are syntactically valid
- **Merge conflict risk:** LOW — isolated to `generate/` directory.
- **Definition of done:** Generator module produces valid SVA properties, coverage analysis works, NL IR is functional.

---

## T7: Description Engine Module

- **Task ID:** T7
- **Task Name:** Description Engine Module
- **Objective:** Port V1's SVAD translator and CoT builder to V3 as `describe/`, rewiring parser dependencies to V3's `sva/` module.
- **Why this task exists:** The description engine provides SVA-to-natural-language translation and chain-of-thought reasoning, used by the dataset builder.
- **Inputs / prerequisites:** T2 (runtime) and T3 (SVA parser) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/src/sva_toolkit/describe/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/describe/translator.py` (from V1 `svad_translator/`)
  - `v3/sva-toolkit/src/sva_toolkit/describe/cot.py` (from V1 `cot_builder/`)
  - `v3/sva-toolkit/tests/describe/`
- **Files to avoid:** `cli/main.py`, `runtime/`, `sva/`, other domain packages.
- **Expected deliverables:**
  - `SVADTranslator` class ported and adapted to use V3's `sva.parser`
  - `SVACoTBuilder` class ported and adapted to use V3's `sva.parser`
  - Template-based translation (no LLM dependency in core path)
  - Unit tests for both translator and CoT builder
- **Suggested validation:**
  - `pytest tests/describe/ -q` — all pass
  - `ruff check src/sva_toolkit/describe/`
  - SVAD output is readable natural language for sample SVA inputs
  - CoT output contains structured reasoning sections
- **Merge conflict risk:** LOW — isolated to `describe/` directory.
- **Definition of done:** Description engine produces SVAD and CoT from SVA code using V3's parser.

---

## T8: Data Workflows Module

- **Task ID:** T8
- **Task Name:** Data Workflows Module
- **Objective:** Port V1's dataset builder and benchmark runner to V3 as `data/`, integrating with the formal and describe modules.
- **Why this task exists:** Data workflows (dataset construction and LLM benchmarking) are key V1 capabilities that depend on multiple other modules.
- **Inputs / prerequisites:** T2 (runtime), T4 (formal), and T7 (describe) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/src/sva_toolkit/data/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/data/dataset.py` (from V1 `dataset_builder/`)
  - `v3/sva-toolkit/src/sva_toolkit/data/benchmark.py` (from V1 `benchmark/`)
  - `v3/sva-toolkit/tests/data/`
- **Files to avoid:** `cli/main.py`, `runtime/`, `sva/`, `formal/`, `describe/`.
- **Expected deliverables:**
  - `DatasetBuilder` class using `runtime.llm` for LLM calls and `describe.cot` for CoT
  - `BenchmarkRunner` class using `runtime.llm` for LLM calls and `formal.service` for verification
  - Multiprocessing support with progress caching
  - Graceful degradation without LLM (offline mode)
  - No hardcoded API keys or model names
  - Unit tests with mocked LLM and formal backends
- **Suggested validation:**
  - `pytest tests/data/ -q` — all pass
  - `ruff check src/sva_toolkit/data/`
  - Dataset builder produces valid JSONL output with mocked LLM
  - Benchmark runner evaluates sample items with mocked backends
- **Merge conflict risk:** LOW — isolated to `data/` directory.
- **Definition of done:** Data workflows are functional with mocked dependencies, multiprocessing works, caching works.

---

## T9: Unified CLI

- **Task ID:** T9
- **Task Name:** Unified CLI
- **Objective:** Build the unified `sva` CLI entry point with subcommand groups for all product domains.
- **Why this task exists:** The CLI is the user-facing interface that ties all modules together under one command.
- **Inputs / prerequisites:** T4, T5, T6, T7, T8 must all be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/src/sva_toolkit/cli/__init__.py`
  - `v3/sva-toolkit/src/sva_toolkit/cli/main.py`
  - `v3/sva-toolkit/src/sva_toolkit/__init__.py` (update version/exports)
  - `v3/sva-toolkit/pyproject.toml` (add `[project.scripts]` entry)
  - `v3/sva-toolkit/tests/cli/`
- **Files to avoid:** Domain module internals (only import their public APIs).
- **Expected deliverables:**
  - `sva` top-level command group
  - `sva parse` — parse SVA and display structure
  - `sva formal check|equivalent|relationship` — formal verification
  - `sva timing render|validate|emit-sva|extract-sva|bundle-sva` — timing diagrams
  - `sva generate` — SVA generation with options for mode, count, coverage
  - `sva describe svad|cot` — SVAD translation and CoT generation
  - `sva data build|benchmark` — dataset building and LLM benchmarking
  - CLI smoke tests for each subcommand
  - `--help` text is clear and consistent across all commands
- **Suggested validation:**
  - `sva --help` shows all subcommand groups
  - `pytest tests/cli/ -q` — all pass
  - Each subcommand responds to `--help`
  - `ruff check src/sva_toolkit/cli/`
- **Merge conflict risk:** MEDIUM — modifies `pyproject.toml` and root `__init__.py`. Only T9 should touch these after T1.
- **Definition of done:** All CLI subcommands are wired, help text is consistent, smoke tests pass.

---

## T10: Documentation & Examples

- **Task ID:** T10
- **Task Name:** Documentation & Examples
- **Objective:** Write comprehensive documentation and port/create examples for all V3 modules.
- **Why this task exists:** Consistent documentation is a stated goal. V2 has minimal docs; V1 has extensive but scattered docs.
- **Inputs / prerequisites:** T9 (CLI) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/README.md`
  - `v3/sva-toolkit/docs/` (new documentation files)
  - `v3/sva-toolkit/examples/` (example scripts and data)
- **Files to avoid:** Source code in `src/` (read-only for doc generation).
- **Expected deliverables:**
  - Comprehensive `README.md` with installation, quickstart, CLI reference
  - Per-module documentation (one markdown file per domain)
  - Architecture overview document
  - Curated examples for each CLI subcommand
  - Example timing diagram DSL files (port from V2)
  - Example SVA files for formal verification
- **Suggested validation:**
  - All example commands in docs actually work
  - No broken internal links
  - `ruff check` still passes (no source changes)
- **Merge conflict risk:** LOW — only touches `docs/` and `examples/`.
- **Definition of done:** Documentation covers all modules, examples are runnable.

---

## T11: Integration Tests & CI Hardening

- **Task ID:** T11
- **Task Name:** Integration Tests & CI Hardening
- **Objective:** Add cross-module integration tests and harden the CI pipeline.
- **Why this task exists:** Unit tests per module don't catch integration issues. CI must gate on lint, test, and build.
- **Inputs / prerequisites:** T9 (CLI) must be DONE.
- **Files or directories likely to be touched:**
  - `v3/sva-toolkit/tests/integration/` (new)
  - `v3/sva-toolkit/.github/workflows/ci.yml` (update)
  - `v3/sva-toolkit/pyproject.toml` (test config tweaks if needed)
- **Files to avoid:** Domain module source code (test-only changes).
- **Expected deliverables:**
  - Integration tests: CLI end-to-end tests, cross-module workflows
  - CI workflow: lint (ruff), type check (mypy optional), test (pytest), build (wheel)
  - Test for: generate → describe → dataset pipeline
  - Test for: timing DSL → SVA → formal check pipeline
  - Test for: CLI subcommand invocation via `click.testing.CliRunner`
- **Suggested validation:**
  - `pytest tests/ -q` — all pass (unit + integration)
  - CI workflow runs successfully
  - `python -m build` produces installable wheel
- **Merge conflict risk:** LOW — primarily new test files.
- **Definition of done:** Integration tests cover key cross-module workflows, CI is green.

---

# 6. Prompt for Each Coding Worker Task

## T1 Worker Prompt

```
You are a coding worker assigned to T1: Project Scaffold.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md

Your task objective:
Create the V3 project directory at v3/sva-toolkit/ with the complete package scaffold.

Dependency check:
- Required completed tasks: None (this is the first task).

Focus areas:
- v3/sva-toolkit/pyproject.toml — model after v2/sva-tools/pyproject.toml but with ALL dependency groups
- v3/sva-toolkit/src/sva_toolkit/ — create __init__.py for every subpackage:
  cli/, sva/, formal/, timing/, generate/, describe/, data/, runtime/
  Also create formal/backends/, sva/lowerings/, timing/core/, timing/frontend/,
  timing/bridge/, timing/projection/, timing/render/, generate/nl/
- v3/sva-toolkit/tests/ — mirror the src structure with __init__.py files
- v3/sva-toolkit/.github/workflows/ci.yml — GitHub Actions for Python 3.11+3.12
- v3/sva-toolkit/Makefile — install, install-dev, test, lint, format, clean, build
- v3/sva-toolkit/.gitignore — comprehensive Python/IDE/OS patterns
- v3/sva-toolkit/README.md — minimal placeholder

pyproject.toml specifics:
- name: sva-toolkit, version: 3.0.0a1
- python: >=3.11
- core deps: click, pydantic
- optional extras:
  - dev: pytest, pytest-asyncio, ruff, mypy
  - llm: openai
  - timing-render: cairosvg, wavedrom
  - formal: (empty, for future external tool deps)
  - rich: rich
  - all: all of the above
- [project.scripts]: sva = "sva_toolkit.cli.main:main"
- [tool.ruff]: target-version = "py311", line-length = 120
- [tool.pytest.ini_options]: pythonpath = ["src"]

Avoid touching unless absolutely necessary:
- v1/ and v2/ directories (read-only reference)
- docs/task_dag_planning.md, docs/task_dag_shared.md

Implementation requirements:
- All __init__.py files should be empty or contain only a version string (root only)
- Root __init__.py: __version__ = "3.0.0a1"
- Do NOT implement any actual module logic — just the skeleton
- CI workflow: checkout, setup-python (3.11, 3.12), pip install -e ".[dev]", ruff check, pytest, python -m build

Validation requirements:
- cd v3/sva-toolkit && pip install -e ".[dev]" succeeds
- ruff check src/ passes
- pytest runs (0 tests collected is OK)
- python -m build produces a wheel

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T2 Worker Prompt

```
You are a coding worker assigned to T2: Runtime Infrastructure.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md

Your task objective:
Port and consolidate the runtime module into v3/sva-toolkit/src/sva_toolkit/runtime/.
Source from v2/sva-tools/src/sva_toolkit/runtime/ (tools, process, config) and
v1/src/sva_toolkit/utils/llm_client.py (LLM client).

Dependency check:
- Required completed tasks: T1 (Project Scaffold)
- If T1 is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/src/sva_toolkit/runtime/config.py — ToolkitConfig, ToolConfig (from V2)
- v3/sva-toolkit/src/sva_toolkit/runtime/tools.py — ToolRegistry, create_default_registry (from V2)
- v3/sva-toolkit/src/sva_toolkit/runtime/process.py — RunResult, run_tool, make_work_dir (from V2)
- v3/sva-toolkit/src/sva_toolkit/runtime/llm.py — NEW: LLMClient, LLMConfig (port from V1 utils/llm_client.py)
- v3/sva-toolkit/src/sva_toolkit/runtime/__init__.py — public exports
- v3/sva-toolkit/tests/runtime/ — unit tests for all components

Avoid touching unless absolutely necessary:
- cli/main.py, sva/, formal/, timing/, generate/, describe/, data/
- pyproject.toml (do not modify)

Implementation requirements:
- Port V2's runtime module as-is (it's clean and well-structured)
- Port V1's LLMClient to runtime/llm.py:
  - LLMConfig dataclass: model, api_key (from env var OPENAI_API_KEY), base_url, temperature, max_tokens
  - LLMClient: __init__(config), generate(system_prompt, user_prompt) -> str
  - Use lazy import of openai (only import when generate() is called)
  - Raise clear error if openai not installed: "Install sva-toolkit[llm] for LLM support"
  - No hardcoded API keys or model names
- All models use dataclasses (matching V2 style), not pydantic for runtime internals
- ToolRegistry.register() should NOT call shutil.which at import time — only when register() is called

Validation requirements:
- pytest tests/runtime/ -q — all pass
- ruff check src/sva_toolkit/runtime/
- LLM tests use unittest.mock to mock openai calls

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T3 Worker Prompt

```
You are a coding worker assigned to T3: SVA Parser Module.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md

Your task objective:
Port V2's custom SVA parser module from v2/sva-tools/src/sva_toolkit/sva/ to
v3/sva-toolkit/src/sva_toolkit/sva/, along with all its tests.

Dependency check:
- Required completed tasks: T1 (Project Scaffold)
- If T1 is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/src/sva_toolkit/sva/ — ALL files:
  __init__.py, lexer.py, parser.py, ast.py, emitter.py, errors.py,
  analysis.py, transforms.py, visitors.py, lowerings/__init__.py, lowerings/conditions.py
- v3/sva-toolkit/tests/sva/ — ALL test files from v2/sva-tools/tests/sva/:
  test_parser.py, test_emitter.py, test_roundtrip.py, test_visitors.py, test_condition_lowering.py

Avoid touching unless absolutely necessary:
- cli/main.py, runtime/, formal/, timing/, generate/, describe/, data/
- pyproject.toml

Implementation requirements:
- This is a direct port from V2 — the SVA parser is already clean and well-structured
- Copy all source files preserving the exact module structure
- Copy all test files preserving the exact test structure
- Ensure __init__.py exports match V2's exports
- No modifications to the parser logic unless fixing obvious bugs
- Verify all internal imports use the correct package path (sva_toolkit.sva.*)

Validation requirements:
- pytest tests/sva/ -q — all V2 tests pass in V3
- ruff check src/sva_toolkit/sva/
- Roundtrip test: parse → emit → parse produces identical AST

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T4 Worker Prompt

```
You are a coding worker assigned to T4: Formal Verification Module.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md

Your task objective:
Port V2's formal verification module from v2/sva-tools/src/sva_toolkit/formal/ to
v3/sva-toolkit/src/sva_toolkit/formal/, along with all its tests.

Dependency check:
- Required completed tasks: T2 (Runtime Infrastructure), T3 (SVA Parser Module)
- If any dependency is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/src/sva_toolkit/formal/ — ALL files:
  __init__.py, model.py, parse.py, normalize.py, service.py,
  backends/__init__.py, backends/ebmc.py, backends/vcformal.py
- v3/sva-toolkit/tests/formal/ — ALL test files from v2/sva-tools/tests/formal/

Avoid touching unless absolutely necessary:
- cli/main.py, runtime/ (import only), sva/ (import only)
- timing/, generate/, describe/, data/

Implementation requirements:
- Direct port from V2's formal module
- FormalService must use runtime.tools.ToolRegistry for backend discovery
- FormalService must use runtime.process.run_tool for subprocess execution
- FormalProperty and CheckResult models use V3's sva.ast types
- parse.py uses V3's sva.parser for property parsing
- Backends (EBMC, VCFormal) are adapters that generate SV modules and invoke external tools
- Ensure all imports reference sva_toolkit.sva.* and sva_toolkit.runtime.* (V3 paths)

Validation requirements:
- pytest tests/formal/ -q — all pass
- ruff check src/sva_toolkit/formal/
- Unit tests mock external tool invocations (no real EBMC/VCFormal needed)

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T5 Worker Prompt

```
You are a coding worker assigned to T5: Timing Diagrams Module.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md

Your task objective:
Port V2's timing diagram module from v2/sva-tools/src/sva_toolkit/timing/ to
v3/sva-toolkit/src/sva_toolkit/timing/, along with all its tests.

Dependency check:
- Required completed tasks: T2 (Runtime Infrastructure), T3 (SVA Parser Module)
- If any dependency is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/src/sva_toolkit/timing/ — ALL files and subdirectories:
  __init__.py, errors.py,
  core/ (scenario.py, conditions.py, graph.py),
  frontend/ (parser.py, validate.py),
  bridge/ (emit_sva.py, from_sva.py, to_dsl.py, solver.py, ebmc_witness.py),
  projection/ (scenario_view.py, wavedrom_view.py),
  render/ (svg.py, png.py, wavedrom.py, waveform.py)
- v3/sva-toolkit/tests/timing/ — ALL test files from v2/sva-tools/tests/timing/

Avoid touching unless absolutely necessary:
- cli/main.py, runtime/ (import only), sva/ (import only)
- formal/, generate/, describe/, data/

Implementation requirements:
- Direct port from V2's timing module
- The timing bridge modules (from_sva, ebmc_witness) depend on formal.model.FormalProperty
  and sva.parser — ensure imports reference V3 paths
- Render modules with optional deps (cairosvg, wavedrom) must use lazy imports
  with clear error messages: "Install sva-toolkit[timing-render] for PNG/WaveDrom support"
- Preserve all V2 test files and fixtures

Validation requirements:
- pytest tests/timing/ -q — all pass
- ruff check src/sva_toolkit/timing/
- Tests that require cairosvg/wavedrom should be skipped if not installed (pytest.importorskip)

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T6 Worker Prompt

```
You are a coding worker assigned to T6: SVA Generator Module.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md
- v1/docs/sva-gen.md (V1 generator documentation)

Your task objective:
Port V1's SVA generator from v1/src/sva_toolkit/gen/ to v3/sva-toolkit/src/sva_toolkit/generate/,
adapting it to V3's architecture, coding standards (Python 3.11+, ruff, type hints), and
runtime infrastructure.

Dependency check:
- Required completed tasks: T2 (Runtime Infrastructure), T3 (SVA Parser Module)
- If any dependency is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/src/sva_toolkit/generate/ — port and adapt:
  __init__.py, synthesizer.py (from gen/generator.py), types.py (from gen/types_sva.py),
  coverage.py, templates.py, utils.py, stratified.py,
  nl/__init__.py, nl/ir.py, nl/extractor.py, nl/realizer.py, nl/symbolic.py, nl/templates.py
- v3/sva-toolkit/tests/generate/ — port from v1/tests/test_full_properties.py and create new tests

Avoid touching unless absolutely necessary:
- cli/main.py, runtime/ (import only), sva/ (import only)
- formal/, timing/, describe/, data/

Implementation requirements:
- Rename gen/ to generate/ for clarity
- Rename generator.py to synthesizer.py (SVASynthesizer class)
- Rename types_sva.py to types.py
- Replace hardcoded Verible paths with runtime.tools.ToolRegistry lookups
- Replace subprocess calls with runtime.process.run_tool
- Remove the 20k-line signal_pool_expanded.py — replace with:
  - A curated signal_presets.py with ~200 representative signals
  - A scripts/expand_signal_pool.py generation script (port from v1/scripts/)
- Use Python 3.11+ syntax: match statements where appropriate, | union types, etc.
- Add proper type hints throughout
- Clean up dataclass definitions (use @dataclass or pydantic where appropriate)
- NL IR module: port as-is but ensure clean imports

Validation requirements:
- pytest tests/generate/ -q — all pass
- ruff check src/sva_toolkit/generate/
- SVASynthesizer can generate N properties and report coverage stats
- Generated SVA is syntactically valid (test with Verible if available, skip if not)

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T7 Worker Prompt

```
You are a coding worker assigned to T7: Description Engine Module.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md
- v1/docs/sva-cot.md (V1 CoT documentation)

Your task objective:
Port V1's SVAD translator and CoT builder to v3/sva-toolkit/src/sva_toolkit/describe/,
rewiring all parser dependencies from V1's Verible-based ast_parser to V3's sva/ module.

Dependency check:
- Required completed tasks: T2 (Runtime Infrastructure), T3 (SVA Parser Module)
- If any dependency is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/src/sva_toolkit/describe/ — port and adapt:
  __init__.py, translator.py (from v1 svad_translator/translator.py),
  cot.py (from v1 cot_builder/builder.py)
- v3/sva-toolkit/tests/describe/ — port from v1/tests/test_cot_builder.py and create new tests

Avoid touching unless absolutely necessary:
- cli/main.py, runtime/ (import only), sva/ (import only)
- formal/, timing/, generate/, data/

Implementation requirements:
- SVADTranslator: port the template-based translation logic from V1's translator.py (966 lines)
  - Replace imports of sva_toolkit.ast_parser with sva_toolkit.sva
  - The translator uses SVA structure (signals, operators, temporal patterns) to generate
    natural language — adapt the structure access to V3's AST types
  - Keep the core translation templates intact
  - No LLM dependency in the core translation path
- SVACoTBuilder: port from V1's cot_builder/builder.py
  - Replace SVAASTParser with V3's sva.parser (parse_property_text or equivalent)
  - Adapt structure field access to V3's AST node types
  - Keep template matching logic intact
  - CoTSection dataclass for structured output
- Both classes should accept raw SVA code strings as input (parse internally)

Validation requirements:
- pytest tests/describe/ -q — all pass
- ruff check src/sva_toolkit/describe/
- SVADTranslator produces readable NL for sample SVA properties
- SVACoTBuilder produces structured reasoning with sections

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- The translator is 966 lines — port it methodically, section by section.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T8 Worker Prompt

```
You are a coding worker assigned to T8: Data Workflows Module.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md
- v1/docs/sva-dataset.md and v1/docs/sva-benchmark.md (V1 documentation)

Your task objective:
Port V1's dataset builder and benchmark runner to v3/sva-toolkit/src/sva_toolkit/data/,
integrating with V3's formal, describe, and runtime modules.

Dependency check:
- Required completed tasks: T2 (Runtime), T4 (Formal), T7 (Describe)
- If any dependency is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/src/sva_toolkit/data/ — port and adapt:
  __init__.py, dataset.py (from v1 dataset_builder/builder.py),
  benchmark.py (from v1 benchmark/runner.py)
- v3/sva-toolkit/tests/data/ — port and adapt tests

Avoid touching unless absolutely necessary:
- cli/main.py, runtime/ (import only), formal/ (import only), describe/ (import only)
- sva/, timing/, generate/

Implementation requirements:
- DatasetBuilder (from V1 dataset_builder/builder.py):
  - Use runtime.llm.LLMClient for LLM calls (not direct openai imports)
  - Use describe.cot.SVACoTBuilder for CoT generation
  - Use describe.translator.SVADTranslator as fallback for non-LLM SVAD
  - Multiprocessing with progress caching (port V1's caching logic)
  - No hardcoded API keys, model names, or file paths
  - Graceful offline mode: if no LLM config, skip SVAD generation and only produce CoT
- BenchmarkRunner (from V1 benchmark/runner.py):
  - Use runtime.llm.LLMClient for LLM calls
  - Use formal.service.FormalService for implication checking
  - Multiprocessing with progress caching
  - RelationshipType enum for result classification
  - No hardcoded paths or API keys
- Both must be testable with mocked LLM and formal backends

Validation requirements:
- pytest tests/data/ -q — all pass
- ruff check src/sva_toolkit/data/
- DatasetBuilder produces valid output structure with mocked LLM
- BenchmarkRunner evaluates items correctly with mocked formal backend

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T9 Worker Prompt

```
You are a coding worker assigned to T9: Unified CLI.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md
- v2/sva-tools/src/sva_toolkit/cli/main.py (V2's CLI as reference)

Your task objective:
Build the unified `sva` CLI entry point that wires all V3 domain modules into a
single command-line interface with consistent subcommand groups.

Dependency check:
- Required completed tasks: T4 (Formal), T5 (Timing), T6 (Generate), T7 (Describe), T8 (Data)
- If any dependency is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/src/sva_toolkit/cli/main.py — the unified CLI
- v3/sva-toolkit/src/sva_toolkit/cli/__init__.py
- v3/sva-toolkit/src/sva_toolkit/__init__.py — update exports if needed
- v3/sva-toolkit/pyproject.toml — ensure [project.scripts] has: sva = "sva_toolkit.cli.main:main"
- v3/sva-toolkit/tests/cli/ — CLI smoke tests

Avoid touching unless absolutely necessary:
- Domain module internals (only import their public APIs)
- runtime/ internals

Implementation requirements:
- Top-level: @click.group() main() with version option
- Subcommand groups and commands:
  sva parse <sva_code_or_file> [--format json|text] — parse SVA, display structure
  sva formal check <antecedent> <consequent> [--backend auto|ebmc|vcformal] [--timeout] [--depth]
  sva formal equivalent <sva1> <sva2> [--backend] [--timeout] [--depth]
  sva formal relationship <sva1> <sva2> [--backend] [--timeout] [--depth]
  sva timing render <input.td> [-o output] [--format svg|png]
  sva timing validate <input.td>
  sva timing emit-sva <input.td> [-o output] [--allow-lossy]
  sva timing extract-sva <input.sv> [-o output] [--depth] [--timeout]
  sva timing bundle-sva <input1.sv> ... [-o output]
  sva generate [--count N] [--mode random|stratified] [--validate] [--coverage]
  sva describe svad <sva_code_or_file> [--format text|json|markdown]
  sva describe cot <sva_code_or_file> [--format text|json|markdown]
  sva data build <input.json> [-o output.jsonl] [--model MODEL] [--workers N]
  sva data benchmark <dataset.json> [--model MODEL] [--workers N] [-o results.json]
- Use lazy imports for all domain modules (import inside command functions)
- Consistent error handling: catch domain exceptions, print user-friendly messages
- All commands support --help with clear descriptions
- Model V2's CLI patterns (see v2/sva-tools/src/sva_toolkit/cli/main.py)

Validation requirements:
- sva --help shows all subcommand groups
- Each subcommand responds to --help
- pytest tests/cli/ -q — smoke tests pass using click.testing.CliRunner
- ruff check src/sva_toolkit/cli/

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T10 Worker Prompt

```
You are a coding worker assigned to T10: Documentation & Examples.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md

Your task objective:
Write comprehensive documentation and create examples for all V3 modules.

Dependency check:
- Required completed tasks: T9 (Unified CLI)
- If T9 is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/README.md — comprehensive with installation, quickstart, CLI reference
- v3/sva-toolkit/docs/ — per-module documentation:
  - architecture.md — system overview, module relationships, data flow
  - sva-parse.md — SVA parser usage and API
  - sva-formal.md — formal verification usage
  - sva-timing.md — timing diagram system
  - sva-generate.md — SVA generator usage
  - sva-describe.md — SVAD and CoT usage
  - sva-data.md — dataset building and benchmarking
- v3/sva-toolkit/examples/ — runnable examples:
  - Port V2's timing diagram examples (td/ and sva/ directories)
  - Create example scripts for each CLI subcommand
  - Sample input files for dataset builder and benchmark

Avoid touching unless absolutely necessary:
- Source code in src/ (read-only — do not modify implementation)
- tests/ (do not modify tests)

Implementation requirements:
- README.md structure: badges, description, installation, quickstart, CLI reference table,
  module overview, development setup, contributing
- Each doc file: purpose, usage examples, API reference, CLI commands
- Reference V1's docs for content (v1/docs/*.md) but rewrite for V3's architecture
- All example commands must use the `sva` unified CLI
- No secrets or real API keys in examples — use placeholders

Validation requirements:
- All CLI commands shown in docs actually work (test manually or note which require external tools)
- No broken internal links between docs
- Examples directory has a README explaining each example

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

## T11 Worker Prompt

```
You are a coding worker assigned to T11: Integration Tests & CI Hardening.
Before doing anything, read:
- docs/task_dag_planning.md (sections 1–2 for project context)
- docs/task_dag_shared.md

Your task objective:
Add cross-module integration tests and harden the CI pipeline for V3.

Dependency check:
- Required completed tasks: T9 (Unified CLI)
- If T9 is not DONE, do not proceed. Record the blocker in docs/task_dag_shared.md.

Focus areas:
- v3/sva-toolkit/tests/integration/ — new integration test directory
- v3/sva-toolkit/.github/workflows/ci.yml — update CI workflow
- v3/sva-toolkit/pyproject.toml — test config tweaks if needed

Avoid touching unless absolutely necessary:
- Domain module source code (test-only changes)
- Existing unit tests in tests/sva/, tests/formal/, etc.

Implementation requirements:
- Integration tests using click.testing.CliRunner:
  - test_cli_formal.py: formal check/equivalent/relationship with mocked backends
  - test_cli_timing.py: timing render/validate/emit-sva with sample .td files
  - test_cli_generate.py: generate command produces valid output
  - test_cli_describe.py: describe svad/cot produces output for sample SVA
  - test_cli_data.py: data build/benchmark with mocked LLM and formal
- Cross-module workflow tests:
  - test_pipeline_generate_describe.py: generate SVA → describe it → verify output
  - test_pipeline_timing_formal.py: timing DSL → emit SVA → formal check (mocked)
- CI workflow enhancements:
  - Matrix: Python 3.11, 3.12
  - Steps: checkout, install, ruff check, ruff format --check, pytest -q, python -m build
  - Optional: mypy (non-blocking)
  - Separate job for integration tests (may need longer timeout)

Validation requirements:
- pytest tests/ -q — all tests pass (unit + integration)
- CI workflow runs successfully in local simulation (act or manual)
- python -m build produces installable wheel
- pip install dist/*.whl && sva --help works

Rules:
- Keep changes minimal and merge-friendly.
- Do not perform unrelated refactors.
- Before finishing, update the task status table in docs/task_dag_shared.md and append an entry to the update log.
- If incomplete, record what is done, what remains, blockers, and next steps.
```

---

# 7. Recommended Execution Order

## First Wave (Sequential — Foundation)
1. **T1: Project Scaffold** — Must complete before anything else. Creates the directory structure and package config that all workers need.

## Second Wave (Parallel — Infrastructure)
2. **T2: Runtime Infrastructure** + **T3: SVA Parser Module** — Run in parallel. These are the two foundational modules that all domain tasks depend on. No conflict risk (different directories).

## Third Wave (Parallel — Domain Modules)
3. **T4: Formal Verification** + **T5: Timing Diagrams** + **T6: SVA Generator** + **T7: Description Engine** — All four run in parallel. Each creates its own subpackage. No conflict risk.

## Fourth Wave (Sequential — Cross-Domain)
4. **T8: Data Workflows** — Depends on T4 (formal) and T7 (describe). Cannot start until both are done.

## Fifth Wave (Sequential — Integration)
5. **T9: Unified CLI** — Depends on all domain modules. Wires everything together.

## Sixth Wave (Parallel — Hardening)
6. **T10: Documentation & Examples** + **T11: Integration Tests & CI** — Run in parallel after CLI is complete.

---

# 8. Open Risks and Manager Notes

## Risk 1: V1 → V3 Parser Adaptation (T6, T7)
The SVA generator (T6) and description engine (T7) were built against V1's Verible-based `ast_parser` module. V3 uses V2's custom parser with different AST types. Workers on T6 and T7 will need to map V1's `SVAStructure`, `Signal`, `ImplicationType`, `TemporalOperator`, `BuiltinFunction` types to V3's `sva.ast` equivalents. This is the highest-risk adaptation work.

**Mitigation:** T6 and T7 workers should study both V1's `ast_parser/parser.py` and V3's `sva/ast.py` before starting. If the mapping is too complex, they should create a thin adapter layer in their module rather than modifying the `sva/` module.

## Risk 2: Signal Pool Size (T6)
V1's `signal_pool_expanded.py` is ~20,000 lines. The task spec says to replace it with curated presets. If the generator's quality depends heavily on the full pool, the worker may need to keep it as package data instead.

**Mitigation:** T6 worker should test generation quality with the curated preset first. If insufficient, include the full pool as a data file loaded at runtime.

## Risk 3: LLM-Dependent Tests (T8)
Dataset builder and benchmark runner require LLM calls. Tests must use mocks, but the mocking surface is large (multiprocessing + LLM + formal backends).

**Mitigation:** T8 worker should focus on single-process mode tests first, then add multiprocessing tests.

## Risk 4: SVAD Translator Complexity (T7)
The SVAD translator is 966 lines of template-based translation logic tightly coupled to V1's AST types. Porting it to V3's AST types is labor-intensive.

**Mitigation:** T7 worker should consider a phased approach: first get the translator working with a subset of SVA patterns, then expand coverage.

## Risk 5: Timing Module Dependencies on Formal (T5)
V2's timing bridge modules (`from_sva.py`, `ebmc_witness.py`) import from `formal.model`. T5 runs in parallel with T4. If T5 finishes before T4, the timing bridge imports will fail.

**Mitigation:** T5 worker should stub the formal imports if T4 is not yet done, or coordinate with T4 worker. Alternatively, the manager can sequence T4 before T5 if workers are limited.

## Risk 6: pyproject.toml as Hot File
T1 creates it, T9 modifies it (CLI entry point). No other task should touch it. If a worker needs a new dependency, they should record it in the shared doc for T9 to add.

**Coordination rule:** Only T1 and T9 may modify `pyproject.toml`. Other workers record dependency needs in the update log.

## General Notes for the Manager
- Launch T1 first and wait for completion before launching any other task.
- T2 and T3 can launch simultaneously as soon as T1 is done.
- T4–T7 can all launch simultaneously as soon as T2 and T3 are done.
- T8 has the most dependencies — it's the critical path bottleneck in Wave 4.
- T9 is the integration point — review its output carefully before launching T10/T11.
- Consider assigning the most experienced worker to T6 or T7 (highest adaptation complexity).
- Total estimated worker-hours: T1 (1h), T2 (2h), T3 (2h), T4 (3h), T5 (3h), T6 (5h), T7 (5h), T8 (4h), T9 (3h), T10 (3h), T11 (3h).
