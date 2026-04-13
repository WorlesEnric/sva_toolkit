# Timing Diagram MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new timing subsystem that parses a small timing-diagram DSL, builds a shared core timing model, renders SVG diagrams, and emits parameterized SVA for a controlled subset.

**Architecture:** The implementation uses one semantic center, `Core Timing Model`, and two thin output projections. The first output path renders SVG timing diagrams from sampled lane values plus semantic overlays. The second output path lowers supported rules into parameterized SVA templates without introducing a second heavy IR.

**Tech Stack:** Python 3.9+, `click`, `dataclasses`, stdlib XML/string generation, `pytest`

---

### Task 1: Create the subsystem skeleton

**Files:**
- Create: `src/sva_toolkit/timing/__init__.py`
- Create: `src/sva_toolkit/timing/core/model.py`
- Create: `src/sva_toolkit/timing/frontend/parser.py`
- Create: `src/sva_toolkit/timing/frontend/validate.py`
- Create: `src/sva_toolkit/timing/projection/diagram_view.py`
- Create: `src/sva_toolkit/timing/projection/assertion_view.py`
- Create: `src/sva_toolkit/timing/render/svg.py`
- Create: `src/sva_toolkit/timing/render/png.py`
- Create: `src/sva_toolkit/timing/bridge/emit_sva.py`
- Create: `src/sva_toolkit/timing/cli/main.py`

**Step 1: Write the failing import smoke test**

Add a test that imports the timing package and checks the public API exists.

**Step 2: Run test to verify it fails**

Run: `pytest tests/timing/unit/test_parser.py -v`

Expected: import error because the package does not exist yet.

**Step 3: Write minimal implementation**

Create the package tree and minimal exports.

**Step 4: Run test to verify it passes**

Run: `pytest tests/timing/unit/test_parser.py -v`

Expected: PASS

### Task 2: Implement the core timing model

**Files:**
- Modify: `src/sva_toolkit/timing/core/model.py`
- Test: `tests/timing/unit/test_parser.py`

**Step 1: Write failing tests for model-backed parsing**

Cover:

- diagram name
- clock definition
- tick count
- params
- lane samples
- event definitions
- rule definitions

**Step 2: Run the targeted tests**

Run: `pytest tests/timing/unit/test_parser.py -v`

Expected: FAIL with missing fields or parse errors.

**Step 3: Write the model classes**

Use dataclasses and a small set of explicit rule/event types. Keep the shape narrow.

**Step 4: Run tests**

Run: `pytest tests/timing/unit/test_parser.py -v`

Expected: PASS

### Task 3: Implement parser and semantic validation

**Files:**
- Modify: `src/sva_toolkit/timing/frontend/parser.py`
- Modify: `src/sva_toolkit/timing/frontend/validate.py`
- Test: `tests/timing/unit/test_parser.py`

**Step 1: Write failing tests for DSL parsing and validation**

Cover:

- duplicate lane name
- unknown signal in event
- lane sample count mismatch
- malformed rule syntax

**Step 2: Run tests**

Run: `pytest tests/timing/unit/test_parser.py -v`

Expected: FAIL with missing validation behavior.

**Step 3: Implement a line-oriented parser**

Support:

- `diagram ... {`
- `clock posedge clk;`
- `disable iff ...;`
- `ticks N;`
- `param NAME;`
- `lane sig: bit = ...;`
- `lane sig: bus = ...;`
- `event name = ...;`
- `rule name: ...;`

**Step 4: Add semantic validation**

Check names, lane references, rule/event references, and sample counts.

**Step 5: Run tests**

Run: `pytest tests/timing/unit/test_parser.py -v`

Expected: PASS

### Task 4: Implement SVG rendering

**Files:**
- Modify: `src/sva_toolkit/timing/projection/diagram_view.py`
- Modify: `src/sva_toolkit/timing/render/svg.py`
- Test: `tests/timing/unit/test_svg_backend.py`

**Step 1: Write failing rendering tests**

Cover:

- SVG root emitted
- lane labels present
- bit waveform paths present
- bus boxes and labels present
- event marker labels present

**Step 2: Run tests**

Run: `pytest tests/timing/unit/test_svg_backend.py -v`

Expected: FAIL because renderer is missing.

**Step 3: Implement a simple layout**

Use fixed tick width and lane height. Support top tick labels and event marker lines.

**Step 4: Run tests**

Run: `pytest tests/timing/unit/test_svg_backend.py -v`

Expected: PASS

### Task 5: Implement parameterized SVA emission

**Files:**
- Modify: `src/sva_toolkit/timing/projection/assertion_view.py`
- Modify: `src/sva_toolkit/timing/bridge/emit_sva.py`
- Test: `tests/timing/unit/test_sva_bridge.py`

**Step 1: Write failing SVA emission tests**

Cover:

- response rule lowers to ranged delay
- hold-until rule lowers to `until_with`
- disable iff and clocking appear in generated property

**Step 2: Run tests**

Run: `pytest tests/timing/unit/test_sva_bridge.py -v`

Expected: FAIL because emitter is missing.

**Step 3: Implement lowering**

Emit one property per rule with stable naming and symbolic parameters preserved.

**Step 4: Run tests**

Run: `pytest tests/timing/unit/test_sva_bridge.py -v`

Expected: PASS

### Task 6: Add CLI entry points

**Files:**
- Modify: `src/sva_toolkit/timing/cli/main.py`
- Modify: `pyproject.toml`
- Test: `tests/timing/integration/test_diagram_to_svg.py`
- Test: `tests/timing/integration/test_diagram_to_param_sva.py`

**Step 1: Write failing CLI integration tests**

Cover:

- render DSL file to SVG text
- emit DSL file to SVA text

**Step 2: Run tests**

Run: `pytest tests/timing/integration -v`

Expected: FAIL because CLI script is missing.

**Step 3: Implement CLI**

Add `sva-diagram` entry point with subcommands:

- `render`
- `emit-sva`
- `validate`

**Step 4: Run tests**

Run: `pytest tests/timing/integration -v`

Expected: PASS

### Task 7: Final targeted verification

**Files:**
- Test: `tests/timing/unit/test_parser.py`
- Test: `tests/timing/unit/test_svg_backend.py`
- Test: `tests/timing/unit/test_sva_bridge.py`
- Test: `tests/timing/integration/test_diagram_to_svg.py`
- Test: `tests/timing/integration/test_diagram_to_param_sva.py`

**Step 1: Run the timing test subset**

Run: `pytest tests/timing -v`

Expected: all timing tests PASS.

**Step 2: Run a smoke import on the package**

Run: `python -c "import sva_toolkit; import sva_toolkit.timing"`

Expected: no exception.

**Step 3: Record follow-up work**

Document deferred items:

- shared SVA AST extraction from `gen/types_sva.py`
- richer grammar support
- PNG export dependency wiring
- sampled capture semantics

