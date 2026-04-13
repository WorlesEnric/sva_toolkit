# Broad-Support SVA <-> Diagram Design

## Goal

Extend the timing subsystem from a one-way pipeline:

```text
diagram DSL -> semantic model -> diagram renderer / SVA emitter
```

into a dual-direction system:

```text
diagram DSL <-> shared timing scenario IR <-> SVA
                                     |
                                     -> symbolic / concrete diagram renderer
```

The key requirement is not full formal invertibility of all SVA. The requirement
is broad practical support for `SVA -> diagram`, with explicit handling for
lossy cases and explicit rejection for properties that do not map to a useful
diagram.

## Assumptions

- Broad SVA coverage is more important than strict round-trip equivalence for
  every property.
- Some SVA inputs should return `unsupported` rather than force a misleading
  diagram.
- The current concrete timing DSL remains valuable and should not be discarded.
- The new system must support grouped diagrams for related properties sharing a
  clock/reset/signal bundle.

## Demand Analysis

### User-facing demands

1. A user can author a timing diagram and emit SVA.
2. A user can input SVA and obtain a readable timing diagram when the property
   has a meaningful temporal scenario.
3. A set of related SVAs can be merged into one diagram when they share a clock,
   reset, and a coherent signal bundle.
4. Infinite or symbolic time must be representable. The diagram cannot require
   enumerating every cycle.
5. The output must be explicit about what is exact, what is heuristic, and what
   is unsupported.

### Engineering demands

1. Keep one canonical semantic center. Do not maintain one concrete-trace model
   for diagrams and a separate temporal skeleton for SVA.
2. Preserve backward compatibility for current `.tdg`-style concrete diagrams.
3. Separate semantic meaning from display hints. Reverse mapping from SVA cannot
   always recover a unique waveform trace.
4. Make unsupported cases a first-class outcome instead of hiding them behind
   vague or wrong pictures.
5. Build a testable extraction pipeline with stable intermediate objects.

## Current State Analysis

The current timing model is a finite sampled-trace model:

- [`LaneSpec.samples`](/Users/wangqihang/wkspace/sva_toolkit/src/sva_toolkit/timing/core/model.py#L35)
  stores one concrete sample per tick.
- [`DiagramSpec.ticks`](/Users/wangqihang/wkspace/sva_toolkit/src/sva_toolkit/timing/core/model.py#L117)
  requires a finite width.
- The parser expects `lane ... = sample sample sample;`
  in [`parser.py`](/Users/wangqihang/wkspace/sva_toolkit/src/sva_toolkit/timing/frontend/parser.py#L31).
- Rules are limited to three direct diagram-friendly forms:
  `not before`, ranged response, and hold-until.

This works for `diagram -> SVA`, but it does not work as the canonical model for
`SVA -> diagram`.

The repository already contains SVA infrastructure that should be reused:

- [`SVAASTParser`](/Users/wangqihang/wkspace/sva_toolkit/src/sva_toolkit/ast_parser/parser.py#L530)
  extracts clock, reset, signals, implication, delays, and some temporal
  operators.
- [`gen/types_sva.py`](/Users/wangqihang/wkspace/sva_toolkit/src/sva_toolkit/gen/types_sva.py)
  defines a typed SVA node system used by the generator.
- [`SVADTranslator`](/Users/wangqihang/wkspace/sva_toolkit/src/sva_toolkit/svad_translator/translator.py#L451)
  contains a lightweight sequence parser for delays, repetition, sequence
  binary ops, and some wrappers.

These pieces are useful, but none of them is currently the right canonical IR
for diagram generation.

## Problem Analysis

### 1. Semantic mismatch: trace vs property

The current DSL describes one finite trace. SVA describes constraints over
infinite sets of traces. One SVA property usually has many valid witness
diagrams. Therefore `SVA -> current DSL` is underdetermined by design.

### 2. Missing symbolic time

SVA commonly contains:

- `##[m:n]`
- `##[m:$]`
- `[*m:n]`
- `until`, `until_with`, `throughout`, `intersect`

These are not “N explicit ticks”. They are symbolic windows or temporal
relations. The current DSL has no representation for omitted time, ranged gaps,
or infinite future/history.

### 3. Missing scenario segmentation

The user requirement for vertical wavy separators is correct. Reverse-mapped
diagrams need explicit timeline cuts:

- omitted prefix history
- symbolic waiting windows
- omitted suffix future
- compressed lookback regions for `$past`

The current DSL has no concept of a cut, symbolic gap, or compressed segment.

### 4. Missing distinction between semantic facts and display choices

From SVA, we can usually recover:

- clock/reset
- trigger conditions
- outcomes
- delays
- hold/stability/change constraints

We usually cannot recover:

- exact prehistory
- exact posthistory
- exact bus values outside constrained points
- one unique lane waveform when multiple waveforms satisfy the property

Therefore the language needs a place for semantic constraints and a separate
place for display hints or derived representative waveforms.

### 5. Grouped-property composition is not free

Merging several SVAs into one diagram is useful but risky. Related properties
can still conflict visually or semantically. A bundler must at least check:

- same clock edge and clock signal
- compatible `disable iff`
- overlapping signal bundle
- compatible anchor ordering
- no impossible contradictory lane constraints in the same displayed scenario

### 6. Reverse support needs confidence and failure modes

The reverse direction needs three explicit outcomes:

- `exact`: the extracted scenario is in the formally supported subset
- `lossy`: the diagram captures the main timing skeleton but not full semantics
- `unsupported`: do not render a diagram

Without this, the renderer will eventually draw precise-looking but wrong
pictures.

## Alternatives Considered

### Option A: Patch the current DSL in place with more samples syntax

Examples: ellipsis markers inside `lane = ...`, range annotations inline, more
rule types.

Pros:

- smallest parser delta
- keeps the current surface feel

Cons:

- still centered on explicit samples
- poor fit for symbolic windows and grouped properties
- mixes formal timing semantics with ad hoc waveform art
- becomes hard to parse and harder to validate

Conclusion: reject as the main design.

### Option B: Create a separate reverse-only DSL

Pros:

- clean design for symbolic scenarios
- no compatibility constraints

Cons:

- two unrelated diagram languages in one subsystem
- weak long-term maintainability
- impossible to talk about “dual direction mapping” cleanly

Conclusion: reject unless backward compatibility becomes impossible.

### Option C: Introduce a DSL v2 superset with a new canonical scenario IR

Pros:

- keeps the current concrete DSL as a valid subset
- supports both concrete traces and symbolic scenarios
- gives one semantic center for both directions
- allows exact and lossy reverse outputs without lying

Cons:

- requires a real refactor
- needs a new IR and parser evolution

Conclusion: recommended.

## Recommended Design

### Decision 1: The canonical model becomes a Timing Scenario IR

Do not use the current `DiagramSpec` as the semantic center anymore. Keep it as
the v1 concrete view. Introduce a new IR that can represent both concrete traces
and symbolic scenarios.

Recommended IR concepts:

- `ScenarioDocument`
  - name
  - clocking
  - params
  - signals
  - anchors
  - windows
  - cuts
  - lane_constraints
  - properties
  - bundle metadata
  - extraction status

- `SignalDecl`
  - name
  - kind: bit/bus
  - width

- `Anchor`
  - name
  - predicate or derived origin
  - role: trigger / response / state / lookback / synthetic

- `TimeWindow`
  - name
  - start anchor
  - end anchor
  - bound: exact / range / unbounded / omitted
  - inclusive semantics

- `Cut`
  - name
  - placement: before anchor / after anchor / between windows
  - meaning: omitted history / omitted future / compressed symbolic gap
  - optional label

- `LaneConstraint`
  - signal
  - region reference
  - relation:
    - `level(0|1|x|z)`
    - `eq(value)`
    - `neq(value)`
    - `stable`
    - `change`
    - `rise`
    - `fall`
    - `unknown`
    - `dontcare`

- `PropertyOverlay`
  - source property name
  - original SVA
  - status: exact / lossy
  - related anchors/windows/constraints
  - notes for unsupported fragments

### Decision 2: DSL v2 is a superset of the current timing DSL

Keep current v1 constructs valid:

- `ticks N;`
- `lane sig: bit = ...;`
- `event name = ...;`
- current `rule ...` forms

Add symbolic constructs for v2:

- signal declarations without samples
- anchors
- windows
- cuts
- lane display/semantic constraints
- explicit property blocks or overlays
- bundle/group metadata

### Decision 3: Reverse mapping emits DSL v2, not v1

`SVA -> diagram DSL` should produce DSL v2 by default. If a scenario happens to
be concrete enough to degrade to v1, that can be an optimization later, but it
must not be the primary target.

### Decision 4: Exactness is a property of the extraction, not the renderer

The renderer should consume scenario IR plus extraction metadata. It should not
decide whether a diagram is exact or lossy.

## DSL v2 Shape

The DSL should remain line-oriented and reviewable in git. A representative
shape:

```text
diagram axi_wait {
  clock posedge ACLK;
  disable iff !ARESETn;

  param READY_WAIT_MAX;
  param RESP_OK;

  lane valid: bit;
  lane ready: bit;
  lane addr: bus[32];
  lane id: bus[4];
  lane resp: bus[2];

  anchor wait_start = high(valid) and low(ready);
  anchor fire = high(valid) and high(ready);
  anchor resp_ok = eq(resp, RESP_OK);

  window wait_gap = between wait_start and fire [0:READY_WAIT_MAX];
  cut prefix = before wait_start omitted;
  cut suffix = after fire omitted;

  show valid = 0 before wait_start;
  show valid = 1 from wait_start until fire;
  show ready = 0 in wait_gap;
  show ready = 1 at fire;
  show stable({addr, id}) in wait_gap;
  show eq(resp, RESP_OK) at resp_ok;

  property ready_after_wait:
    wait_start |-> ##[0:READY_WAIT_MAX] fire;

  property ok_resp_after_fire:
    fire |-> ##0 resp_ok;

  property addr_stable_until_fire:
    wait_start |-> stable({addr, id}) until_with fire;
}
```

### Notes on the new syntax

- `lane` without `= ...` means “declared for symbolic display”.
- `anchor` is a named event/state point.
- `window` is a named temporal interval with explicit bounds.
- `cut` renders as a vertical wavy separator or compressed region marker.
- `show` is display-oriented but still typed and validated.
- `property` stores source semantics and supports grouped output.

## Mapping Strategy: SVA -> Scenario IR

### Supported broad subset

The reverse extractor should target the following first-class constructs:

- `@(posedge/negedge clk)`
- `disable iff (...)`
- `|->`, `|=>`
- `##n`, `##[m:n]`, `##[m:$]`
- `$rose`, `$fell`, `$stable`, `$changed`
- simple predicates: `sig`, `!sig`, `sig == value`, `sig != value`
- `until`, `until_with`
- `throughout`
- `intersect`
- `[*]`, `[*m:n]`, `[=]`, `[->]`
- simple property `and`, `or`
- simple `$past(sig, n)` as lookback annotation

### Mapping table

| SVA construct | Scenario IR |
|---|---|
| `@(posedge clk)` | `clock edge=posedge signal=clk` |
| `disable iff (rst)` | document disable condition |
| antecedent of implication | trigger anchor |
| consequent base condition | response anchor or lane constraint |
| `|=>` | implicit exact one-cycle window before consequent |
| `##n` | exact window |
| `##[m:n]` | ranged symbolic window |
| `##[m:$]` | unbounded symbolic window + cut |
| `$rose(a)` | anchor predicate `rise(a)` |
| `$fell(a)` | anchor predicate `fall(a)` |
| `$stable(a)` | lane constraint `stable` over region |
| `$changed(a)` | lane constraint `change` at anchor or region |
| `expr until_with evt` | hold window ending at event anchor |
| `expr throughout seq` | lane constraint over extracted window of `seq` |
| `$past(a, n)` | lookback anchor/window relative to current anchor |

### Lossy constructs

These should be accepted, but the extractor should mark the output `lossy`:

- property `and` or `or` across multiple timing skeletons
- nested sequences with more than one plausible scenario layout
- `if/else` property forms
- mixed data-path arithmetic with temporal skeleton
- `intersect` when it creates competing interval layouts

### Unsupported constructs

These should return `unsupported` unless a later implementation adds explicit
support:

- local variables or sampled-value variables requiring dataflow state
- `accept_on`, `reject_on`, `sync_accept_on`, `sync_reject_on`
- heavily nested `first_match` with competing branches
- properties whose main content is arithmetic/logical without a meaningful
  timing skeleton
- cases where extracted anchors/windows are contradictory

## Grouping Multiple SVAs into One Diagram

### Grouping criteria

Properties may be bundled only if all of the following hold:

1. same clock signal and edge
2. same normalized `disable iff`
3. signal overlap above a threshold
4. extracted anchor graph is compatible
5. no direct contradiction on displayed lane constraints

Recommended clustering strategy:

- primary key: `(clock_edge, clock_signal, disable_iff_normalized)`
- secondary grouping: signal-overlap graph with Jaccard threshold, start with
  `0.35`
- split groups when:
  - more than `10` displayed signals
  - more than `5` properties
  - more than `3` independent timing chains

### Bundle conflict rules

A bundle must be split if two properties require incompatible displayed facts in
the same region, for example:

- `show ready = 0 in wait_gap`
- `show ready = 1 in same wait_gap`

unless the contradiction is isolated to different alternative branches and the
renderer explicitly supports branch views. That is not recommended for phase 1.

## Renderer Requirements for DSL v2

The renderer must stop assuming that every x-axis unit is one explicit tick.
It needs three timeline primitives:

1. concrete ticks
2. symbolic windows with labels like `[0:MAX]`
3. cuts / omitted regions rendered as vertical wavy separators or compressed
   bands

Recommended rendering behavior:

- concrete v1 diagrams keep today’s explicit sample layout
- ranged or unbounded windows render as compressed labeled regions
- omitted prefix/suffix/history render as cuts
- property overlays refer to anchors and windows, not raw tick indices
- grouped properties show a legend or footer summary with exact/lossy status

## Exact Round-Trip Policy

The system should not promise `SVA -> DSL v2 -> SVA` equivalence for the full
language. Instead:

- For an `exact` extraction subset, the emitted DSL v2 should lower back to an
  equivalent supported SVA skeleton.
- For `lossy` extraction, lowering back to SVA is optional and must be blocked
  or explicitly marked as heuristic.

This is necessary to avoid accidental generation of false formal claims from a
diagram that was only meant as a visualization.

## Proposed Module Changes

### New or refactored core modules

- `src/sva_toolkit/timing/core/scenario.py`
  - canonical scenario IR
- `src/sva_toolkit/timing/core/conditions.py`
  - shared predicate / boolean condition objects
- `src/sva_toolkit/timing/bridge/from_sva.py`
  - SVA -> scenario extraction
- `src/sva_toolkit/timing/bridge/to_dsl.py`
  - scenario -> DSL v2 emission
- `src/sva_toolkit/timing/projection/scenario_view.py`
  - render-oriented view from symbolic scenario IR

### Existing modules to evolve

- `src/sva_toolkit/timing/frontend/parser.py`
  - support v2 constructs while keeping v1 syntax
- `src/sva_toolkit/timing/frontend/validate.py`
  - validate symbolic windows, cuts, and `show` clauses
- `src/sva_toolkit/timing/bridge/emit_sva.py`
  - lower exact v2 subset
- `src/sva_toolkit/timing/render/wavedrom.py`
  - render symbolic windows and cuts, not only explicit samples

### Existing repo modules to reuse

- `src/sva_toolkit/ast_parser/parser.py`
  - source of clock/reset/signal extraction and raw Verible AST
- `src/sva_toolkit/svad_translator/translator.py`
  - sequence splitting heuristics that can be lifted into the extractor
- `src/sva_toolkit/gen/types_sva.py`
  - long-term candidate for a richer typed normalized AST

## Improvement Plan

### Phase 1: Introduce the canonical scenario IR

Deliverables:

- scenario dataclasses
- extraction status model: `exact`, `lossy`, `unsupported`
- uplift adapter from current `DiagramSpec` to scenario IR

Exit criteria:

- current concrete DSL can still render and emit SVA through scenario IR

### Phase 2: Define and parse DSL v2

Deliverables:

- grammar for `lane` declarations without samples
- `anchor`, `window`, `cut`, `show`, `property`
- validation rules for symbolic timelines

Exit criteria:

- parser supports both v1 concrete diagrams and v2 symbolic diagrams
- invalid symbolic scenarios fail with precise diagnostics

### Phase 3: Build SVA -> scenario extraction

Deliverables:

- extractor from `SVAStructure` + raw property body
- normalization of implications, delays, repeats, until/throughout forms
- diagramability classifier

Exit criteria:

- targeted SVA corpus yields `exact`, `lossy`, or `unsupported`
- no silent fallback to fake waveforms

### Phase 4: Add property bundling

Deliverables:

- grouping by clock/reset
- signal-overlap clustering
- conflict detection
- bundle split heuristics

Exit criteria:

- groups are deterministic
- contradictory scenarios are split instead of merged

### Phase 5: Extend the renderer

Deliverables:

- symbolic windows and cuts
- lookback / omitted history markers
- bundle legend / footer
- exact/lossy badges in output metadata

Exit criteria:

- broad reverse-mapped examples render without pretending symbolic regions are
  explicit ticks

### Phase 6: Exact subset lowering

Deliverables:

- formal lowering from exact v2 subset back to SVA
- explicit block on lossy re-lowering unless user opts in

Exit criteria:

- exact round-trip tests pass for the supported subset

## Testing Strategy

### Unit tests

- parser tests for v2 constructs
- validation tests for windows, cuts, and `show`
- extractor tests: `SVA -> status + scenario IR`
- bundling tests
- renderer tests for symbolic gaps and cuts

### Golden tests

Store goldens for:

- input SVA
- extraction status JSON
- emitted DSL v2
- rendered SVG fragments

### Corpus categories

1. exact single-chain handshake properties
2. ranged latency properties
3. hold/stability properties
4. repetition properties
5. lossy but still diagrammable compound properties
6. unsupported properties

## Risks and Mitigations

### Risk: ambiguous extraction produces misleading diagrams

Mitigation:

- extraction status model
- unsupported outcomes
- footer notes for lossy fragments

### Risk: DSL v2 becomes too large and ad hoc

Mitigation:

- keep a strict split between semantic declarations and display clauses
- require every new construct to map to scenario IR, not directly to SVG

### Risk: grouping logic becomes unstable

Mitigation:

- deterministic clustering
- small bundle size caps
- split on conflicts early

### Risk: parser complexity explodes

Mitigation:

- keep line-oriented syntax
- avoid expression-general grammar in v2
- parse SVA richness in the extractor, not in the DSL surface

## Recommended Implementation Order

If implementing this now, the order should be:

1. scenario IR
2. v1 uplift into scenario IR
3. DSL v2 parser/validator
4. SVA extractor with `exact/lossy/unsupported`
5. renderer support for windows and cuts
6. bundle grouping
7. exact subset re-lowering to SVA

## Summary

The right move is not “emit the current diagram DSL from SVA”. The right move
is:

1. define a new canonical timing scenario IR
2. evolve the current timing DSL into a symbolic v2 superset
3. emit DSL v2 from SVA
4. classify reverse results as exact, lossy, or unsupported
5. keep exact round-tripping only for the supported subset

This preserves the current concrete authoring path while making `SVA -> diagram`
broad, honest, and production-viable.
