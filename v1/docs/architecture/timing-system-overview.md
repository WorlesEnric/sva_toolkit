# Timing System Overview

## Goal

Build a single-core timing system that serves two outputs from one source of truth:

1. Render hardware timing diagrams from a diagram DSL as SVG first and PNG optionally.
2. Emit parameterized SVA templates from the same timing semantics.

The intended pipeline is:

```text
diagram DSL
  -> parser
  -> normalizer / elaborator
  -> Core Timing Model
  -> Diagram Projection -> layout -> SVG -> PNG
  -> Assertion Projection -> parameterized SVA
```

## Design Principles

### Single semantic core

The system has exactly one heavy intermediate representation: `Core Timing Model`.
Everything else is a thin projection or backend-specific view.

### Semantic-first, not pixel-first

The DSL describes clocking, lanes, events, captures, and constraints. It does not
directly describe SVG paths or SVA syntax.

### SVG-first rendering

SVG is the primary rendering target because timing diagrams are vector graphics.
PNG is a derived export format.

### Phased implementation

The first milestone intentionally implements only a controlled subset:

- single clock domain
- optional `disable iff`
- bit lanes and bus lanes, including explicit widths such as `bus[32]`
- sampled lane values across discrete ticks
- event predicates:
  - `rise`, `fall`, `high`, `low`
  - `eq(signal, value)`
  - `change(signal)`
  - `stable(signal)`
  - grouped stability via `stable({a, b, c})`
  - conjunction of the supported predicates
- basic constraints:
  - `not A before B`
  - `A -> after [MIN:MAX] B`
  - `<predicate_expr> from A until B`
- SVG rendering
- parameterized SVA emission for the supported constraints

## Scope Boundaries

### In scope for MVP

- parse a compact, line-oriented DSL
- build a typed core model
- render realistic-looking bit and bus lanes
- render event markers and simple response overlays
- emit parameterized SVA text for supported rule kinds
- validate the DSL at semantic level

### Out of scope for MVP

- multi-clock timing
- async sampling semantics
- arbitrary arithmetic expressions
- complex sequence operators such as `first_match`
- full SVA AST unification with `gen/types_sva.py`
- globally optimal annotation routing
- required PNG conversion dependency bundled in the package

## Core Timing Model

The core model is the semantic center of the subsystem.

### Top-level object

`DiagramSpec`

- `name`
- `clocking`
- `ticks`
- `params`
- `lanes`
- `events`
- `rules`

### Clocking

`ClockingSpec`

- `edge`: `posedge` or `negedge`
- `signal`: clock signal name
- `disable_iff`: optional reset / disable expression

### Parameters

`ParameterDecl`

- `name`
- `kind`: currently `int` or `bits`

Parameters remain symbolic in emitted SVA.

### Lanes

`LaneSpec`

- `name`
- `kind`: `bit` or `bus`
- `width`: optional symbolic or integer width
- `samples`: one value per tick

Bit lanes use values like `0`, `1`, `x`, `z`.
Bus lanes use symbolic values such as `A`, `ID0`, `READ`, `x`.

### Events

`EventSpec`

- `name`
- `expr`

The event expression subset for MVP:

- `rise(sig)`
- `fall(sig)`
- `high(sig)`
- `low(sig)`
- `eq(sig, VALUE)`
- `change(sig)`
- `stable(sig)`
- `stable({a, b, c})`
- `high(a) and high(b) same_cycle`
- `low(a) and high(b) same_cycle`

### Rules

`RuleSpec` is one of:

- `NotBeforeRule`
- `ResponseRule`
- `HoldUntilRule`

These are semantic rules, not visual instructions. The renderer may visualize them
as markers, arrows, or highlights. The SVA backend lowers them into properties.

## DSL Shape

The first DSL is intentionally simple and regular:

```text
diagram req_ack {
  clock posedge clk;
  disable iff !rst_n;
  ticks 5;

  param LAT_MIN;
  param LAT_MAX;

  lane req: bit = 0 1 1 1 0;
  lane ack: bit = 0 0 0 1 0;

  event req_start = rise(req);
  event ack_seen = rise(ack);

  rule no_early_ack:
    not ack_seen before req_start;

  rule ack_after_req:
    req_start -> after [LAT_MIN:LAT_MAX] ack_seen;

  rule req_hold_until_ack:
    high(req) from req_start until ack_seen;
}
```

An extended multibit example:

```text
diagram axi_wait {
  clock posedge ACLK;
  disable iff !ARESETn;
  ticks 6;

  param READY_WAIT_MAX;
  param RESP_OK;

  lane valid: bit = 0 1 1 1 1 0;
  lane ready: bit = 0 0 0 1 1 0;
  lane addr: bus[32] = x A0 A0 A0 A0 x;
  lane id: bus[4] = x 3 3 3 3 x;
  lane resp: bus[2] = xx xx xx OK OK xx;

  event wait_start = high(valid) and low(ready) same_cycle;
  event fire = high(valid) and high(ready) same_cycle;
  event resp_ok = eq(resp, RESP_OK);
  event addr_shift = change(addr);

  rule ready_after_wait:
    wait_start -> after [0:READY_WAIT_MAX] fire;

  rule ok_resp_after_fire:
    fire -> after [0:0] resp_ok;

  rule addr_stable_until_fire:
    stable({addr, id}) from wait_start until fire;
}
```

The syntax is kept line-oriented so that parsing and diagnostics stay simple in the
first implementation.

## Module Layout

```text
src/sva_toolkit/timing/
  frontend/
    ast.py
    parser.py
    normalize.py
    validate.py
  core/
    model.py
    expressions.py
  projection/
    diagram_view.py
    assertion_view.py
  render/
    scene.py
    layout.py
    svg.py
    png.py
  bridge/
    emit_sva.py
  cli/
    main.py
```

### Responsibilities

- `frontend`: text parsing, normalization, semantic validation
- `core`: shared timing semantics
- `projection`: thin adapters from core model to backend-oriented views
- `render`: geometric layout and drawing
- `bridge`: parameterized SVA emission
- `cli`: command-line entry points

## Rendering Architecture

The rendering path is:

```text
Core Timing Model
  -> DiagramView
  -> WaveDrom source
  -> WaveDrom SVG
  -> timing overlays and summary
  -> SVG
```

The production renderer uses WaveDrom for the waveform layout and lane geometry,
then composes timing-aware overlays on top of that stable base. This keeps the
core timing semantics in one place while avoiding bespoke waveform drawing
logic.

The overlay path is deterministic and greedy:

1. compute event label tracks to avoid text collisions
2. route response windows on dedicated top-band tracks
3. apply hold highlights per affected lane
4. emit a footer rule summary for semantics that are clearer as text

This is intentionally not a global optimization engine. The goal is predictable,
readable output for common protocol diagrams.

## SVA Lowering Rules

### Event lowering

- `rise(sig)` -> `$rose(sig)`
- `fall(sig)` -> `$fell(sig)`
- `high(sig)` -> `sig`
- `low(sig)` -> `!sig`
- `eq(sig, VALUE)` -> `(sig == VALUE)`
- `change(sig)` -> `$changed(sig)`
- `stable(sig)` -> `$stable(sig)`
- conjunction -> `(expr_a && expr_b)`

### Rule lowering

- `not ack_seen before req_start`
  - emitted as a safety property forbidding `ack_seen` until `req_start`
- `req_start -> after [LAT_MIN:LAT_MAX] ack_seen`
  - emitted as `req_start |-> ##[LAT_MIN:LAT_MAX] ack_seen`
- `stable({addr, id}) from wait_start until fire`
  - emitted as `wait_start |-> ($stable(addr) && $stable(id)) until_with fire`

The lowering style is intentionally conservative and human-readable.

## Validation Rules

The semantic validator must reject:

- duplicate names across params, lanes, events, or rules
- undeclared signal references
- event references to unknown lanes
- sample count mismatch with `ticks`
- malformed delay ranges
- invalid use of `high/low` on missing lanes

## Testing Strategy

### Unit tests

- parser tests for supported DSL constructs
- validator tests for name and shape errors
- rendering tests that assert key SVG fragments
- SVA emission tests that assert generated property text

### Golden-style tests

Store small example DSL files and expected SVG/SVA outputs for regression control.

## Evolution Path

### Phase 1

- line DSL
- shared core model
- SVG renderer
- parameterized SVA emitter

### Phase 2

- more event and window forms
- captures and sampled-value comparisons
- richer overlay routing
- optional PNG export integration

### Phase 3

- unify timing SVA lowering with a shared formal SVA AST
- richer protocol libraries
- multi-clock / asynchronous extensions
