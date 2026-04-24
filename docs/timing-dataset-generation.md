# Timing Diagram Dataset Generation Design

## Purpose

This document describes a procedural generator for large Image-DSL datasets for
the timing diagram DSL. The target task is to train a model that recovers
canonical timing DSL code from a rendered sequence/timing diagram image.

The generator should not sample raw DSL text directly. The timing DSL has
semantic cross-references between lanes, anchors, windows, cuts, constraints,
properties, and rendered overlays. A direct grammar sampler will produce many
invalid, trivial, or visually ambiguous examples. The recommended design is a
scenario-first generator:

```text
primitive temporal idioms
  -> temporal event graph
  -> typed signal schema
  -> concrete or symbolic constraints
  -> waveform synthesis
  -> ScenarioDocument
  -> canonical DSL
  -> SVG/PNG image
  -> JSONL metadata record
```

The canonical source of truth should be
`sva_toolkit.timing.core.scenario.ScenarioDocument`. DSL text should be emitted
with `sva_toolkit.timing.bridge.to_dsl.emit_timing_dsl`, then parsed and
validated again before rendering.

## Non-Goals

- The generator does not need hundreds of hand-written protocol templates.
- The generator does not need to prove all generated properties formally.
- The generator should not require one-to-one recovery of invisible information
  such as comments, formatting, or raw property text that does not affect the
  rendered figure.
- The generator should not train on arbitrary raw DSL variants unless the
  rendered image contains enough visual evidence to recover them.

## Key Principle

Use a small library of primitive temporal idioms and compose them with a graph
generator. Full protocol-like scenarios emerge from composition, not from a
large manually curated motif set.

For example, the same three-node topology can represent many protocols:

```text
A -> B -> C

req_rise      -> ack_rise      -> done_rise
aw_valid      -> aw_handshake  -> b_valid
tx_start      -> rx_first      -> credit_return
irq_assert    -> sw_clear      -> irq_deassert
cmd_accept    -> data_last     -> resp_valid
desc_fetch    -> dma_start     -> dma_done
```

The graph shape is reused, while signal roles, predicates, constraints, bounds,
waveforms, names, widths, and visual complexity are independently varied.

## Dataset Target

Each generated item should produce at least:

```json
{
  "id": "td_000001",
  "seed": 12345,
  "dsl_path": "dsl/td_000001.td",
  "svg_path": "images/td_000001.svg",
  "png_path": "images/td_000001.png",
  "features": {
    "topology": "chain",
    "idioms": ["response", "hold_until", "stable_while"],
    "ticks": 12,
    "lane_count": 7,
    "anchor_count": 4,
    "window_count": 3,
    "has_bus": true,
    "has_params": true,
    "bound_kinds": ["range", "parameter"],
    "constraint_regions": ["from_until", "before", "at"]
  },
  "target": {
    "canonical_dsl": "diagram ...",
    "recoverability": "visual"
  }
}
```

The canonical target should be the emitted DSL, not the original sampled text.
This avoids training the model to infer arbitrary formatting or comments that
are not visible in the diagram.

## Generator Architecture

### 1. Structure Generator

The structure generator creates a temporal event graph. Nodes become anchors.
Edges become timing windows.

Supported topologies should include:

| Topology | Shape | Use |
| --- | --- | --- |
| `single_response` | `A -> B` | req/ack, irq/clear, valid/ready |
| `chain` | `A -> B -> C` | multi-phase transactions |
| `fork` | `A -> B`, `A -> C` | one request with multiple responses |
| `join` | `A -> C`, `B -> C` | two prerequisites before completion |
| `parallel` | `A -> B`, `C -> D` | independent activity in one diagram |
| `burst` | `first -> beat... -> last` | packet/data burst behavior |
| `backpressure` | `valid_rise -> handshake` | stall followed by acceptance |
| `setup_hold` | `setup -> sample -> hold_end` | stability around sample point |

The graph must be acyclic for simple tick assignment. Cyclic protocol behavior
can still be represented by unrolling it into multiple beat anchors.

Recommended internal representation:

```python
@dataclass(frozen=True)
class EventNode:
    id: str
    role: str
    predicate_kind: str | None = None
    absolute_tick: int | None = None


@dataclass(frozen=True)
class TemporalEdge:
    id: str
    start: str
    end: str
    min_delay: int | str
    max_delay: int | str
    bound_kind: str
```

### 2. Semantic Decorator

The semantic decorator assigns signal roles, predicates, and constraints to the
abstract graph.

It should choose a domain flavor, then allocate names from role pools.

Example role pools:

| Flavor | Control Signals | Bus Signals |
| --- | --- | --- |
| generic | `req`, `ack`, `valid`, `ready`, `done` | `addr`, `data`, `id` |
| axi_like | `AWVALID`, `AWREADY`, `WVALID`, `BVALID` | `AWADDR`, `WDATA`, `BRESP` |
| fifo | `push`, `pop`, `full`, `empty` | `wdata`, `rdata`, `level` |
| noc | `TX_VALID`, `RX_READY`, `CR_VALID` | `TX_HDR`, `RX_DATA`, `VC` |
| dma | `desc_valid`, `desc_ready`, `done` | `desc_addr`, `length`, `status` |
| interrupt | `irq`, `clear`, `mask`, `pending` | `cause`, `status` |
| memory | `cmd_valid`, `cmd_ready`, `rsp_valid` | `addr`, `wdata`, `rdata` |

Predicate choices should be type-aware:

| Predicate | Valid Signal Kind | Example |
| --- | --- | --- |
| `rise` | bit | `rise(req)` |
| `fall` | bit | `fall(busy)` |
| `high` | bit | `high(valid)` |
| `low` | bit | `low(error)` |
| `change` | bit or bus | `change(state)` |
| `stable` | bit or bus | `stable(data)` |
| `eq` | bit or bus | `eq(state, BUSY)` |
| `neq` | bit or bus | `neq(resp, ERR)` |

Conjunctions should be generated frequently because they are common in timing
diagrams:

```text
high(VALID) and high(READY)
rise(VALID) and eq(OPCODE, READ)
high(BVALID) and high(BREADY) and eq(BRESP, OKAY)
```

### 3. Primitive Idiom Library

The generator only needs a small idiom library. Idioms are composable semantic
operations over the event graph.

#### Response

Represents a bounded response from one anchor to another.

```text
window response = between trigger and response [min:max];
property p_response:
  trigger |-> ##[min:max] response;
```

Variation axes:

- exact bound: `[2:2]`
- range bound: `[1:5]`
- parameterized bound: `[0:MAX_LAT]`
- unbounded bound: `[1:$]`
- overlapping or delayed response semantics
- response anchor as `rise`, `high`, conjunction, or equality

#### Hold Until

Represents a signal that must hold from a start anchor until an end anchor.

```text
show high(valid) from valid_rise until handshake;
show stable(addr) from valid_rise until handshake;
```

Variation axes:

- held relation: `high`, `low`, `stable`, `eq`
- one lane or multiple lanes
- control-only, bus-only, or mixed control/bus hold
- short hold, long hold, or parameterized hold

#### Stable While

Represents bus stability inside a window or from one anchor to another.

```text
show stable(data) in data_window;
show stable(addr) from launch until capture;
```

Variation axes:

- one bus lane or a bundle of bus lanes
- stable in a named window versus from/until anchors
- values visible in concrete samples or symbolic-only lanes

#### Not Before

Represents an ordering guard.

```text
show low(resp_valid) before req_handshake;
property p_not_before:
  !resp_valid until req_handshake;
```

Variation axes:

- forbidden event before anchor
- forbidden event before window end
- signal low before start
- no response before request acceptance

#### Backpressure

Represents valid asserted while ready is low, then a handshake.

```text
anchor valid_rise = rise(valid);
anchor handshake = high(valid) and high(ready);
window wait = between valid_rise and handshake [0:READY_MAX];
show high(valid) from valid_rise until handshake;
```

Variation axes:

- zero-cycle ready
- one-cycle stall
- long stall
- multiple independent channels
- stable metadata during stall

#### Burst

Represents a first beat, middle beats, and last beat.

```text
anchor first = high(valid) and high(ready);
anchor last = high(valid) and high(ready) and high(last);
window burst = between first and last [0:MAX_BEATS];
show high(valid) from first until last;
show stable(id) from first until last;
```

Variation axes:

- fixed or variable length
- explicit `last` signal or count equality
- payload changes per beat while ID remains stable
- response after last beat

#### Setup/Hold

Represents stability around a sampling event.

```text
window setup = between launch and sample [SETUP:SETUP];
window hold = between sample and hold_end [0:HOLD];
show stable(data) from launch until hold_end;
```

Variation axes:

- pre-sample setup only
- post-sample hold only
- combined setup and hold
- clock edge flavor: `posedge` or `negedge`

#### Cut / Omitted Region

Represents hidden timeline regions.

```text
cut pre_transaction = before req_rise omitted label "idle";
cut post_response = after done omitted label "next transaction";
cut middle_gap = between w1 and w2 compressed label "traffic";
```

Variation axes:

- before anchor
- after anchor
- between windows
- omitted, compressed, or lookback meaning
- short labels, long labels, or no labels

### 4. Constraint Generator

Constraints should be generated after the event graph and signal schema exist.
This allows type checking and cross-reference validation before DSL emission.

Supported regions:

| Region | DSL Shape | Example |
| --- | --- | --- |
| `at` | `show EXPR at A;` | `show high(valid) at capture;` |
| `in` | `show EXPR in W;` | `show stable(data) in transfer;` |
| `before` | `show EXPR before A;` | `show low(resp) before req_hs;` |
| `after` | `show EXPR after A;` | `show high(done) after complete;` |
| `from_until` | `show EXPR from A until B;` | `show stable(addr) from req until ack;` |

The generator should maintain a compatibility matrix:

| Relation | Bit Lane | Bus Lane | Notes |
| --- | --- | --- | --- |
| `high` | yes | no | validator rejects non-bit high |
| `low` | yes | no | validator rejects non-bit low |
| `rise` | yes | no | use mostly for anchors, rarely constraints |
| `fall` | yes | no | use mostly for anchors, rarely constraints |
| `stable` | yes | yes | most useful for buses |
| `eq` | yes | yes | value must be present |
| `neq` | yes | yes | value must be present |

The generator should also avoid semantic contradictions in one region, such as
`show high(valid) from A until B` and `show low(valid) from A until B`.

### 5. Tick Assignment

For concrete examples, assign absolute ticks that satisfy all finite windows.

Recommended policy:

1. Topologically sort the event graph.
2. Assign root anchors to a small offset, usually 1 or 2.
3. For each edge, sample a concrete delay within the finite part of the bound.
4. Set the end anchor to at least `start_tick + delay`.
5. If multiple predecessors point to one anchor, use the maximum proposed tick.
6. Reject and resample if the final tick exceeds the selected tick budget.

For parameterized bounds, use a concrete sampled delay for the rendered trace
but keep the symbolic parameter in the DSL:

```text
window response = between req and ack [1:MAX_LAT];
```

The rendered example may place `ack` at `req + 3`, while the DSL keeps
`MAX_LAT`.

For unbounded bounds, choose a finite visual delay:

```text
window eventually = between start and done [1:$];
```

The concrete waveform still needs a finite `done` tick.

### 6. Waveform Synthesis

Waveform synthesis makes the visual image match the semantic scenario.

Basic algorithm:

1. Initialize bit lanes to `0`.
2. Initialize bus lanes to `x`.
3. Apply anchor predicates at assigned ticks.
4. Apply span constraints over their regions.
5. Fill stable bus regions with deterministic symbolic values.
6. Add distractor activity outside constrained regions.
7. Check that each anchor condition evaluates true at its intended tick.
8. Check that constraints do not conflict.

Examples of waveform operations:

```text
rise(sig) at t:
  sig[t - 1] = 0
  sig[t] = 1

fall(sig) at t:
  sig[t - 1] = 1
  sig[t] = 0

high(sig) from A until B:
  sig[A_tick:B_tick] = 1

stable(bus) from A until B:
  bus[A_tick:B_tick] = generated_value

low(sig) before A:
  sig[0:A_tick] = 0
```

Diversifying visual appearance is important. Add controlled distractors:

- unrelated lanes toggling outside constrained windows
- bus values changing before and after stable regions
- idle prefix or suffix regions
- extra lanes that are not referenced by anchors
- short and long signal names
- varied tick counts
- varied lane order

Distractors must not break anchor predicates or constraints.

### 7. Symbolic-Only Examples

The toolkit can render symbolic scenarios by projecting anchors and windows to a
canonical concrete view. The dataset should include symbolic-only examples, but
they should be used carefully because the image may not encode all original
intent.

Recommended split:

- Concrete examples: primary training set.
- Symbolic examples: auxiliary training or evaluation set.
- Mixed examples: include lanes with samples and lanes without samples.

For symbolic examples, the target should still be canonical emitted DSL.
Metadata should mark `recoverability = "partial_visual"` when details are not
fully visible.

### 8. Property Overlay Generation

Property text is not always visually recoverable unless the renderer exposes it
through overlays or summary rows. For Image-DSL recovery, generated properties
should be limited to visually grounded categories:

| Category | Required Metadata | Visual Evidence |
| --- | --- | --- |
| response | two anchors, one non-omitted window | arrow/span between anchors |
| hold | two anchors, omitted window, from/until constraints | highlighted stable/hold region |
| not_before | two anchors, until-style body | ordering summary/constraint |

Avoid arbitrary raw SVA property bodies in the primary dataset. They create
targets that the model cannot infer from pixels.

If unsupported or lossy properties are included, either render their status and
notes visibly or keep them in a separate split.

### 9. Canonicalization

For a supervised Image-DSL dataset, the target DSL should be canonicalized.
Recommended canonicalization path:

```python
document = generated_scenario_document
dsl = emit_timing_dsl(document)
roundtrip = parse_diagram(dsl)
validate_diagram(roundtrip)
canonical_dsl = emit_timing_dsl(roundtrip)
```

Benefits:

- stable ordering
- stable whitespace
- no invisible comments
- validator-backed target correctness
- easier exact-match evaluation

Do not use randomly formatted DSL as the target unless the task explicitly
requires style recovery.

### 10. Coverage-Guided Sampling

Pure random sampling will overproduce easy cases. The generator should track
coverage buckets and bias toward under-covered features.

Recommended buckets:

| Bucket | Values |
| --- | --- |
| topology | single_response, chain, fork, join, parallel, burst, setup_hold |
| idiom | response, hold_until, stable_while, not_before, backpressure, cut |
| tick_count | 4-6, 7-12, 13-20, 21+ |
| lane_count | 2-3, 4-6, 7-12, 13+ |
| lane_kind | bit_only, bus_only, mixed |
| anchor_count | 1-2, 3-5, 6+ |
| window_count | 0, 1, 2-4, 5+ |
| bound_kind | exact, range, parameterized, unbounded |
| predicate | rise, fall, high, low, change, stable, eq, neq, conjunction |
| region | at, in, before, after, from_until |
| cut | none, before, after, between |
| rendering | concrete, symbolic, mixed |
| naming | short, protocol_like, uppercase, snake_case |

Selection policy:

```text
score(candidate) =
  sum(weight(bucket_value) for newly covered or under-covered features)
  - penalty(for excessive complexity)
  - penalty(for visual ambiguity)
```

Accept high-scoring candidates first. Continue generation until every required
bucket has enough examples.

### 11. Rejection Filters

A candidate should be rejected if any of these checks fail:

- DSL parse fails.
- `validate_diagram` fails.
- SVG/PNG rendering fails.
- Any sampled lane has the wrong number of ticks.
- Any bit lane contains values other than `0`, `1`, `x`, `z`.
- A window references unknown anchors.
- A constraint references an unknown lane, anchor, or window.
- A property references invisible or unsupported semantics in the primary split.
- Rendered image is too small, too large, or visually empty.
- Labels overlap severely.
- The example is a duplicate of an existing DSL or image hash.
- The scenario is trivial when the target split requires moderate complexity.

Recommended duplicate keys:

- canonical DSL hash
- rendered SVG hash after stable normalization
- feature signature
- optional perceptual hash for PNG

### 12. Train/Validation/Test Splits

Use split strategies that measure generalization rather than memorization.

Recommended splits:

| Split | Holdout Strategy |
| --- | --- |
| random | basic sanity check |
| topology-heldout | hold out one graph topology |
| flavor-heldout | hold out one naming/domain flavor |
| bound-heldout | hold out uncommon bound kinds |
| size-heldout | hold out large diagrams |
| rendering-heldout | hold out symbolic or mixed examples |

The model should be evaluated on exact canonical DSL match, parsed AST match,
and semantic equivalence through `parse_diagram`.

### 13. Minimal Implementation Plan

#### Phase 1: Core Generator

Add a timing dataset generator package, for example:

```text
sva_toolkit/timing/generate/
  __init__.py
  model.py
  names.py
  topology.py
  idioms.py
  waveform.py
  coverage.py
  dataset.py
```

Responsibilities:

- `model.py`: internal event graph and sampled scenario objects.
- `names.py`: signal pools, value pools, protocol flavors.
- `topology.py`: DAG/topology generation.
- `idioms.py`: response, hold, stable, not-before, burst, cut decorators.
- `waveform.py`: tick assignment and sample synthesis.
- `coverage.py`: coverage buckets and scoring.
- `dataset.py`: emit DSL, render images, write JSONL.

#### Phase 2: CLI

Add a CLI command:

```bash
sva timing generate-dataset \
  --count 10000 \
  --seed 1 \
  --out dataset/timing \
  --format png \
  --split train
```

Useful flags:

```text
--min-ticks
--max-ticks
--min-lanes
--max-lanes
--concrete-ratio
--symbolic-ratio
--max-retries
--coverage-target
--holdout-topology
--holdout-flavor
```

#### Phase 3: Validation Tests

Add tests that verify:

- generated documents validate
- emitted DSL round-trips
- rendered SVG is non-empty
- bit samples are legal
- all declared samples match `ticks`
- coverage buckets are populated
- fixed seed generation is deterministic

### 14. Pseudocode

```python
def generate_dataset(count: int, seed: int, out_dir: Path) -> None:
    rng = GenerationRng(seed)
    coverage = CoverageTracker()
    records = []

    while len(records) < count:
        for attempt in range(MAX_RETRIES):
            spec = sample_candidate_spec(rng, coverage)

            graph = generate_topology(spec, rng)
            schema = generate_signal_schema(spec, graph, rng)
            decorated = apply_idioms(spec, graph, schema, rng)
            ticks = assign_ticks(decorated, rng)
            samples = synthesize_waveforms(decorated, ticks, rng)

            document = build_scenario_document(decorated, samples)
            dsl = emit_timing_dsl(document)

            try:
                parsed = parse_diagram(dsl)
                validate_diagram(parsed)
                canonical_dsl = emit_timing_dsl(parsed)
                svg = render_diagram_svg(parsed)
            except Exception:
                continue

            if not passes_visual_filters(svg, parsed):
                continue
            if is_duplicate(canonical_dsl, svg):
                continue

            feature_record = extract_features(parsed, spec)
            score = coverage.score(feature_record)
            if not coverage.accept(score, rng):
                continue

            item_id = next_id()
            write_text(out_dir / "dsl" / f"{item_id}.td", canonical_dsl)
            write_text(out_dir / "svg" / f"{item_id}.svg", svg)
            maybe_write_png(parsed, out_dir / "png" / f"{item_id}.png")

            records.append(
                {
                    "id": item_id,
                    "seed": spec.seed,
                    "dsl_path": f"dsl/{item_id}.td",
                    "svg_path": f"svg/{item_id}.svg",
                    "features": feature_record,
                    "target": {"canonical_dsl": canonical_dsl},
                }
            )
            coverage.update(feature_record)
            break
        else:
            raise RuntimeError("failed to generate enough valid timing diagrams")

    write_jsonl(out_dir / "records.jsonl", records)
```

### 15. Example Generated Scenario Families

The following examples show how a small idiom set creates broad variation.

#### Valid/Ready With Stable Metadata

```text
anchor valid_rise = rise(valid);
anchor handshake = high(valid) and high(ready);
window wait = between valid_rise and handshake [0:READY_MAX];
show high(valid) from valid_rise until handshake;
show stable(addr) from valid_rise until handshake;
show stable(id) from valid_rise until handshake;
show low(resp_valid) before handshake;
```

#### Forked Response

```text
anchor req = rise(req);
anchor ack = rise(ack);
anchor irq = rise(irq);
window ack_wait = between req and ack [1:4];
window irq_wait = between req and irq [2:IRQ_MAX];
show high(req) from req until ack;
```

#### Burst With Completion

```text
anchor first = high(valid) and high(ready);
anchor last = high(valid) and high(ready) and high(last);
anchor done = rise(done);
window burst = between first and last [0:MAX_BEATS];
window completion = between last and done [1:RESP_MAX];
show high(valid) from first until last;
show stable(id) from first until last;
```

#### Setup/Hold Capture

```text
anchor launch = rise(enable);
anchor capture = rise(valid);
anchor hold_end = high(done);
window setup = between launch and capture [SETUP:SETUP];
window hold = between capture and hold_end [0:HOLD];
show stable(data) from launch until hold_end;
```

#### Parallel Channels

```text
anchor aw_req = rise(AWVALID);
anchor aw_hs = high(AWVALID) and high(AWREADY);
anchor w_req = rise(WVALID);
anchor w_last = high(WVALID) and high(WREADY) and high(WLAST);
window aw_wait = between aw_req and aw_hs [0:AW_MAX];
window w_phase = between w_req and w_last [0:W_MAX];
show stable(AWADDR) from aw_req until aw_hs;
show high(WVALID) from w_req until w_last;
```

## Recommended Defaults

Initial generation defaults:

| Setting | Default |
| --- | --- |
| ticks | 6 to 20 |
| lanes | 3 to 12 |
| anchors | 2 to 8 |
| windows | 1 to 6 |
| bus lane ratio | 30% to 50% |
| concrete examples | 80% |
| symbolic examples | 10% |
| mixed examples | 10% |
| cuts | 20% to 35% |
| parameterized bounds | 25% |
| unbounded bounds | 5% to 10% |
| distractor lanes | 0 to 3 |
| max retries per accepted item | 100 |

## Evaluation Recommendations

Use multiple metrics:

- Exact canonical DSL match.
- Parsed `ScenarioDocument` structural match.
- Semantic match after ignoring invisible comments and formatting.
- Rendered-image consistency after regenerating the prediction.
- Feature-wise accuracy: lanes, samples, anchors, windows, constraints.

For model training, it is useful to expose the canonical DSL only. For analysis,
store decomposed metadata so errors can be classified by DSL feature.

## Summary

The dataset generator should be compositional, typed, and coverage-guided. A
small primitive idiom library is sufficient if it is combined with temporal
graph generation, semantic role assignment, waveform synthesis, validation,
rendering, and rejection filtering. This produces diverse Image-DSL pairs
without manually inventing hundreds of full scenario templates.
