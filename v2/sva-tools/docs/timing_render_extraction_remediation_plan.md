# Timing Render and Extraction Remediation Plan

## Status

This document captures the currently confirmed timing-render and SVA-extraction problems in `sva-tools`, explains why they matter for industrial use, and proposes a phased remediation plan.

The focus is the path:

1. timing diagram (`.td`)
2. emitted SVA (`.sv`)
3. extracted timing diagram (`gen_td`)
4. rendered SVG (`gen_svg`)

The immediate requirement is to keep WaveDrom as the SVG renderer while fixing incorrect generated waveforms and improving extraction fidelity.

## Executive Summary

Three behaviors must be separated clearly:

1. Exact sample-for-sample roundtrip equality is not guaranteed by the current architecture and should not be treated as a correctness requirement by itself.
2. Incorrect rendered semantics are real bugs. The generated SVG must not violate the meaning of the extracted property, especially for hold, stability, and response behavior.
3. Signal-kind degradation is a real extraction bug. Stable-only buses must not silently become one-bit lanes.

The highest-priority problem is not that `gen_td` lacks samples. The real defect is that the current WaveDrom synthesis path falls back to anchor-local waveform construction and can produce traces that contradict the property semantics represented in the generated SVA.

For industrial use, the tool needs deterministic, reviewable, semantics-preserving concrete traces for sample-less documents. There are two viable strategies:

- internal canonical trace synthesis from the extracted scenario model
- external witness-trace generation using a formal backend such as EBMC or SymbiYosys, followed by deterministic normalization back into timing lanes

Both are acceptable. The short-term implementation with the lowest external dependency is internal canonical trace synthesis. The most formally defensible long-term implementation is witness generation, with deterministic post-processing layered on top.

## Primary Code Areas

The problems described here are not isolated to one module. They span extraction, serialization, parsing, projection, and rendering.

Primary files involved:

- `src/sva_toolkit/cli/main.py`
- `src/sva_toolkit/timing/render/svg.py`
- `src/sva_toolkit/timing/render/wavedrom.py`
- `src/sva_toolkit/timing/projection/wavedrom_view.py`
- `src/sva_toolkit/timing/bridge/to_dsl.py`
- `src/sva_toolkit/timing/frontend/parser.py`
- `src/sva_toolkit/timing/bridge/from_sva.py`
- `src/sva_toolkit/timing/bridge/solver.py`
- `src/sva_toolkit/timing/core/conditions.py`
- `src/sva_toolkit/timing/core/scenario.py`

This matters because an apparently local fix in the renderer will not hold unless the extractor, DSL roundtrip, and semantic model agree on the same invariants.

## Confirmed Problems

### 1. High: WaveDrom rendering of generated symbolic diagrams can be semantically wrong

#### Observed behavior

For generated timing diagrams without explicit samples, the WaveDrom projector synthesizes concrete samples. In at least one reproduced case, the synthesized waveform violates the property intent.

Representative example:

- `examples/td/04_hold_until_ready.td`
- `examples/sva/04_hold_until_ready.sv`
- `examples/gen_td/04_hold_until_ready.td`

The original property describes that `valid` must remain asserted until the handshake with `ready`. The generated WaveDrom rendering can drop `valid` before `ready` rises.

#### Root cause

The failure is structural and has multiple layers:

1. The WaveDrom projector depends on rule classification for hold and response overlays.
2. The emitted timing DSL drops `PropertyOverlay.related_anchors`, `related_windows`, and `related_constraints`.
3. Reparsing the generated `.td` cannot reconstruct those links reliably from the property body text alone.
4. When classification fails, the failure is swallowed and rendering continues.
5. The fallback path synthesizes samples by satisfying anchors only, not span semantics.

As a result, the output may be renderable but semantically wrong.

#### Why this matters

In an industrial flow, a timing diagram is not decoration. It is documentation used in reviews, signoff discussions, IP handoff, post-silicon debug, and protocol validation. A waveform that contradicts the underlying property is more dangerous than a missing waveform because it creates false confidence.

#### Important nuance

Switching symbolic documents to the symbolic SVG renderer would avoid the wrong-waveform bug, but that is not acceptable if WaveDrom output is a hard requirement. Therefore the correct fix is not "avoid WaveDrom". The correct fix is "make sample synthesis semantics-aware".

### 2. High: Current property metadata is not robust enough for WaveDrom classification

#### Observed behavior

Even before the DSL roundtrip, extracted/generated properties are not always shaped the way the current WaveDrom classifier expects.

The current classifier conceptually expects:

- two key anchors: start and end
- optionally one timing window
- optionally one or more hold constraints

However, extracted properties can carry:

- several anchors from the DAG expansion
- several windows from zero-delay structural steps
- no `FROM_UNTIL` constraints for `until_with` forms in the AST/DAG extraction path

That means preserving today's metadata through the DSL is necessary but not sufficient. The classification contract itself is too brittle.

#### Why this matters

If the renderer depends on metadata that is fragile, over-expanded, or not canonicalized, render correctness will continue to regress whenever extraction logic changes.

### 3. Medium: Stable-only bus signals are downgraded to bit lanes

#### Observed behavior

Signals referenced only in `$stable(...)` or equivalent stability constructs are currently inferred as `bit` rather than `bus`.

Representative examples:

- `examples/td/02_data_stable.td` vs `examples/gen_td/02_data_stable.td`
- `examples/td/10_axi4_write_channel.td` vs `examples/gen_td/10_axi4_write_channel.td`

This is why many extracted bus lanes appear as flat one-bit traces in generated WaveDrom output.

#### Root cause

The extractor only marks a signal as bus-like when it sees comparison operators such as `==` or `!=` with a non-trivial right-hand side. Signals used only in stability predicates never become bus candidates.

#### Why this matters

Bus lanes carry protocol meaning:

- address stability
- ID stability
- burst attributes
- response codes
- payload transfer

Reducing them to single-bit lanes destroys both readability and review value.

### 4. Low: Exact waveform equality between `td` and `gen_td` is not a valid requirement

#### Observed behavior

Original timing diagrams may contain explicit samples. SVA emission retains temporal rules and predicates, not the original sample table. Extraction then rebuilds symbolic lanes without original sample values.

#### Interpretation

This behavior is expected under the current representation and should not be classified as a bug on its own.

#### What remains a bug

It is still a bug when:

- the rendered result violates the extracted timing semantics
- bus-ness is lost for signals that should remain buses
- the generated concrete trace is unstable or misleading

## Additional Findings

### 1. The current fallback behavior hides defects

WaveDrom classification failures are currently swallowed and rendering falls back to anchor-only synthesis. This is useful for development convenience, but it is not acceptable as an industrial default because the tool can silently produce incorrect diagrams.

Recommended direction:

- keep best-effort fallback only in explicitly non-strict mode
- add a strict mode that fails rendering when semantics cannot be recovered
- surface warnings whenever rule recovery fails

### 2. Test baseline already indicates instability in the extract path

At review time, the CLI regression covering `extract-sva` example output was already inconsistent with the expected number of properties. This is a sign that the extraction contract is not stable enough and needs clearer invariants and stronger regression coverage.

### 3. WaveDrom can stay the renderer if sample synthesis becomes canonical

There is no architectural requirement to switch away from WaveDrom. The real requirement is to provide it with concrete, deterministic, semantics-preserving samples.

## Industrial Requirements

Any fix should be evaluated against the following requirements.

### 1. Determinism

The same input must produce the same output every time:

- identical `gen_td`
- identical `gen_svg`
- stable lane ordering
- stable anchor naming or stable user-facing labels

This is essential for code review, CI diffs, reproducible documentation, and tool qualification.

### 2. Semantic soundness

Generated concrete samples must satisfy the extracted timing rules:

- hold constraints must hold across the full span
- stability constraints must not be broken by synthesis
- response windows must respect the selected delay policy
- "not before" rules must not be violated

### 3. Explicit handling of under-constrained values

If the source SVA does not determine a unique bus value:

- the tool must not invent a misleading protocol-specific constant
- the tool should either use a deterministic placeholder or mark the value as unknown
- the chosen policy must be documented

### 4. Scalable syntax support

The supported SVA subset must be documented explicitly. Unsupported constructs must be:

- rejected cleanly
- lowered conservatively with a status marker
- or represented as symbolic/raw constraints without pretending to have exact semantics

### 5. Strict and permissive operating modes

Industrial usage usually needs both:

- strict mode for CI, qualification, and signoff
- permissive mode for exploratory documentation workflows

## Recommended Solution Strategy

### Phase 1: Fix signal-kind inference first

This is the smallest, safest, highest-signal improvement.

### Required behavior

- A signal used only in `$stable(data)` must remain a `bus` if it is semantically data-like.
- If width cannot be recovered, preserve `kind=bus` with unknown width rather than forcing `bit`.

### Implementation direction

Improve AST-based signal metadata collection so bus candidates are recognized from:

- equality and inequality against non-trivial values
- `$stable(signal)` on non-control/data-like signals
- potentially `$changed(signal)`, `$past(signal)` comparisons, and other value-carrying uses

### Recommendation

Be conservative:

- do not infer width unless it is known
- do preserve bus kind when the signal is clearly value-bearing

This avoids incorrect precision while preserving readability.

### Phase 2: Make WaveDrom rule recovery independent of fragile metadata

This is the key fix for the render bug.

### Recommendation

Add a rule-recovery layer in the WaveDrom projector that reconstructs renderable semantics from the property bodies themselves when `related_*` metadata is absent, incomplete, or structurally expanded.

### Recovery targets

Recover at least these patterns:

1. response
   - `trigger |-> ##[m:n] response`
   - `trigger |=> ##[m:n] response`

2. hold
   - `trigger |-> hold_expr until_with response`
   - `trigger |-> hold_expr until response`

3. not-before
   - `!expr until anchor_expr`

### Mapping method

Do not rely on object equality of parsed conditions. Normalize to SVA text and compare:

- anchor condition -> normalized SVA text
- property AST subexpression -> normalized SVA text

This provides a stable join key between extracted anchors and parsed property structure.

### Why this is better than relying only on `related_*`

Because it:

- survives DSL roundtrip loss
- survives DAG expansion noise
- works on generated property bodies directly
- decouples render correctness from extraction metadata formatting

### Phase 3: Drive sample synthesis from recovered rules, not just anchors

This phase converts recovered semantics into concrete WaveDrom-ready traces.

### Required behavior

For sample-less documents, sample synthesis must apply span-level semantics:

- hold `high(valid)` from trigger through end
- hold `low(sig)` when required
- hold `eq(bus, value)` across the interval
- for `stable(bus)`, keep a deterministic constant placeholder through the interval

### Recommended synthesis policy

Use canonical trace synthesis, not arbitrary pulse placement.

Suggested defaults:

- trigger edges occur at earliest legal tick
- bounded responses occur at earliest legal response tick by default
- held bit lanes remain asserted for the entire protected interval
- stable bus lanes keep a deterministic placeholder value across the interval
- unconstrained bus lanes outside protected intervals remain `x`

### Deterministic bus placeholder policy

The tool needs a documented policy for stable buses with unknown values.

Reasonable options:

1. all `x`
   - safest, but visually weak

2. stable symbolic placeholder per lane, such as `DATA0`
   - strongest documentation value
   - must be clearly synthetic

3. protocol-looking values like `A0`, `D0`, `ID0`
   - readable, but must not imply that the original value was recovered

Recommended policy:

- use a deterministic synthetic placeholder for stable unknown buses
- ensure the same lane always gets the same placeholder within one document
- document that placeholders are synthesized, not recovered

### Phase 4: Preserve property links in the DSL anyway

Even though renderer correctness should not depend entirely on `related_*`, the DSL should still preserve this metadata.

### Why preserve it

- improves debuggability
- improves symbolic renderer and analysis tools
- enables faster classification
- reduces recomputation

### Recommendation

Extend the property syntax or property-attached metadata format so the following survive roundtrip:

- related anchors
- related windows
- related constraints
- extraction status
- notes

This should be machine-readable, not comment-only.

### Phase 5: Optional witness-trace synthesis backend

For industrial-grade confidence, add a formal witness path in addition to internal canonical synthesis.

### Why witness synthesis is attractive

It uses the formal meaning of the SVA itself to produce a satisfying trace. This is a stronger foundation than a renderer-local heuristic.

### Suitable tools

Potential backends:

- EBMC
- SymbiYosys with cover properties
- other formal engines with witness/VCD support

### Practical note

An external witness still does not uniquely recover the original timing diagram. It only provides one satisfying trace. Therefore witness generation must still be followed by deterministic normalization.

### Recommended role of witness synthesis

Use it as:

- a high-confidence trace backend
- a verification oracle for canonical internal synthesis
- an optional strict mode for signoff-quality generated diagrams

### Why not depend on witness synthesis only

- adds toolchain dependencies
- increases runtime and environment complexity
- may produce different witnesses across solvers or solver versions
- still needs deterministic post-processing to avoid diff churn

## Current vs Target SVA Support

The supported SVA subset should be made explicit and expanded deliberately.

### Current practical support

Relatively tractable today:

- boolean predicates
- `|->`, `|=>`
- `##N`, `##[M:N]`
- basic `until`, `until_with` in some extraction paths
- some repetition forms, often lossy
- `disable iff` at property header level

### Weak or unstable areas

- `throughout`
- `intersect`
- `first_match`
- local sampled variables
- complex branching and merging
- accurate extraction of hold constraints from all AST/DAG paths
- multi-clock properties
- symbolic width recovery

### Target support profile for industrial use

The tool should classify syntax into three buckets:

1. exact
   - semantics preserved fully

2. lossy but valid for visualization
   - semantics preserved approximately and marked explicitly

3. unsupported
   - must not silently render as if exact

Recommended target exact support:

- single-clock implication properties
- bounded delays
- `until_with` hold behavior
- `until` with explicit lossy marker if termination-cycle semantics are approximated
- common stability predicates
- conjunctions of simple predicates
- protocol handshake patterns

Recommended near-term lossy support:

- `throughout`
- selected repetition forms
- limited branching

Recommended unsupported until explicitly implemented:

- `first_match` with non-trivial branching
- local variables with dataflow dependencies
- multi-clock sequencing
- complex control wrappers

### Suggested support matrix

| SVA construct | Current status | Target near-term status | Notes |
| --- | --- | --- | --- |
| `|->`, `|=>` | mostly supported | exact | Must remain exact. |
| `##N`, `##[M:N]` | supported | exact | Canonical response rendering depends on this. |
| `until_with` | unstable across paths | exact | Required for handshake-hold rendering. |
| `until` | partially supported | lossy or exact | Must document termination-cycle semantics. |
| `$stable(sig)` | supported syntactically, weak type inference | exact for extraction and render | Must preserve bus kind. |
| `$changed(sig)` | limited | exact where value-independent | Should not force bit typing for value-carrying signals. |
| `throughout` | weak/lossy | lossy | Acceptable if clearly marked. |
| `intersect` | weak/lossy | unsupported or lossy | Do not pretend exactness. |
| `first_match` | unsupported | unsupported | Must fail or mark unsupported explicitly. |
| repetition `[*]`, `[=]`, `[->]` | mixed/lossy | exact for simple consecutive, lossy otherwise | Must define canonical rendering semantics. |
| `disable iff` | header-level support | exact | Must remain attached to extracted document. |
| local sampled variables | unsupported | unsupported initially | Requires true dataflow modeling. |
| multi-clock properties | unsupported | unsupported initially | Needs a different time model. |

## Canonical Trace Synthesis vs Witness Synthesis

Both approaches are useful. They serve different purposes.

### Canonical internal trace synthesis

Advantages:

- no external tool dependency
- deterministic by construction
- fast
- easy to keep stable in CI

Disadvantages:

- must encode semantics manually
- risks drifting from formal meaning if not validated

### External witness synthesis

Advantages:

- grounded in actual property satisfiability
- easier to justify formally
- valuable for strict validation

Disadvantages:

- requires toolchain integration
- witness is not unique
- still needs deterministic normalization

### Recommended combined architecture

Use internal canonical synthesis as the default renderer path and add witness synthesis as:

- an optional validation mode
- a fallback backend for difficult properties
- a regression oracle during development

## Proposed Work Breakdown

### Workstream A: Extraction fidelity

1. fix bus-kind inference for stable-only and value-bearing symbolic signals
2. document width-unknown bus behavior
3. add focused regression cases from `02_data_stable` and `10_axi4_write_channel`

### Workstream B: WaveDrom semantics recovery

1. add property-body AST parsing inside the WaveDrom projector
2. recover response, hold, and not-before rules from parsed properties
3. map parsed expressions back to extracted anchors using normalized SVA text
4. stop treating missing `related_*` metadata as fatal for correct rendering

### Workstream C: Canonical sample synthesis

1. implement span-aware synthesis for recovered rules
2. choose and document deterministic placeholder policy for unknown bus values
3. ensure overlay generation and lane samples agree

### Workstream D: DSL metadata preservation

1. extend property serialization format
2. parse and validate the metadata
3. preserve links through roundtrip

### Workstream E: Strictness and diagnostics

1. add strict render mode
2. add warnings for recovery failures
3. fail fast when a supposedly exact render cannot be justified

### Workstream F: Optional witness backend

1. generate formal harnesses from extracted scenarios or property bodies
2. request cover witnesses
3. import VCD/witness traces into concrete lane samples
4. normalize witness traces deterministically

## Acceptance Criteria

The remediation should be considered successful only when all of the following are true.

### Render correctness

- `examples/gen_td/04_hold_until_ready.td` renders with `valid` held until the handshake
- hold overlays and rendered samples agree
- render correctness does not depend on preserved `related_*` metadata alone

### Bus fidelity

- stable-only bus signals remain buses after extraction
- generated SVG shows meaningful bus lanes instead of flat one-bit traces

### Determinism

- repeated runs produce identical `gen_td` and `gen_svg`
- placeholder values are stable and documented

### Transparency

- exact, lossy, and unsupported cases are clearly distinguishable
- fallback behavior is visible to users and CI

### SVA subset clarity

- supported syntax is documented
- unsupported constructs are rejected or marked explicitly

## Recommended Immediate Next Steps

1. Fix bus-kind inference for stable-only buses.
2. Implement AST-based rule recovery in the WaveDrom projector.
3. Implement canonical span-aware sample synthesis.
4. Add strict-mode diagnostics and regressions for the known examples.
5. Only after render correctness is stable, extend the DSL to preserve property metadata.
6. Evaluate EBMC or SymbiYosys witness generation as a second-stage enhancement, not as the first unblocker.

## Conclusion

The current issues are not cosmetic. They expose a deeper mismatch between:

- symbolic timing semantics
- extracted property structure
- and concrete WaveDrom sample synthesis

The correct direction is not to abandon WaveDrom. The correct direction is to supply WaveDrom with deterministic, semantics-preserving concrete traces and to make render correctness independent from fragile incidental metadata.

For industrial use, the tool must become explicit about what it knows exactly, what it synthesizes canonically, and what it cannot yet support. Once that contract is clear and enforced, `gen_td` and `gen_svg` can become reliable artifacts rather than best-effort visualizations.
