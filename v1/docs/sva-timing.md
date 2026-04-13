# SVA Timing (`sva-timing`)

## Purpose

`sva-timing` is the command-line interface for the timing-diagram subsystem. It
consumes the timing diagram DSL, validates the specification, renders timing
diagrams, and emits parameterized SystemVerilog Assertions from the same timing
semantics.

This tool is intended for protocol documentation workflows in which the same
timing description must support both human-readable diagrams and formal
assertion templates.

## Capabilities

- Validate timing DSL files
- Render timing diagrams as SVG via a WaveDrom-based backend with semantic overlays
- Export timing diagrams as PNG when `cairosvg` is installed
- Emit parameterized SVA properties from supported timing rules

## Installation

The CLI is installed with the package:

```bash
pip install -e .
```

After installation, the following command becomes available:

```bash
sva-timing --help
```

The older alias `sva-diagram` is currently kept for compatibility, but
`sva-timing` is the preferred command name.

## DSL Input

The tool expects a `.tdg`-style timing DSL file. Example:

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

  rule ready_after_wait:
    wait_start -> after [0:READY_WAIT_MAX] fire;

  rule ok_resp_after_fire:
    fire -> after [0:0] resp_ok;

  rule addr_stable_until_fire:
    stable({addr, id}) from wait_start until fire;
}
```

## Commands

### 1. Validate

Checks syntax and semantic consistency of the DSL.

```bash
sva-timing validate examples/diagram_dsl/axi_wait.tdg
```

Expected output:

```text
valid
```

Validation covers:

- duplicate identifiers
- unknown lane or event references
- invalid predicate/lane combinations
- sample count mismatches against `ticks`

### 2. Render SVG

Renders the DSL file to SVG and writes to stdout unless `--output` is given.
The renderer uses WaveDrom for waveform geometry and adds timing-specific
annotations for events, response windows, hold regions, and a rule summary.

```bash
sva-timing render examples/diagram_dsl/axi_wait.tdg > axi_wait.svg
```

Or explicitly:

```bash
sva-timing render examples/diagram_dsl/axi_wait.tdg --output axi_wait.svg
```

### 3. Render PNG

Renders the DSL file to PNG. This requires the optional `cairosvg` dependency.

```bash
sva-timing render examples/diagram_dsl/axi_wait.tdg --format png --output axi_wait.png
```

If `cairosvg` is not installed, the tool reports an error and suggests using SVG
instead.

### 4. Emit Parameterized SVA

Lowers supported timing rules into parameterized SVA properties.

```bash
sva-timing emit-sva examples/diagram_dsl/axi_wait.tdg
```

Example output:

```systemverilog
property p_ready_after_wait(int READY_WAIT_MAX);
  @(posedge ACLK) disable iff (!ARESETn)
    (valid && !ready) |-> ##[0:READY_WAIT_MAX] (valid && ready);
endproperty

property p_ok_resp_after_fire;
  @(posedge ACLK) disable iff (!ARESETn)
    (valid && ready) |-> ##[0:0] (resp == RESP_OK);
endproperty

property p_addr_stable_until_fire;
  @(posedge ACLK) disable iff (!ARESETn)
    (valid && !ready) |-> ($stable(addr) && $stable(id)) until_with (valid && ready);
endproperty
```

## Supported DSL Features

### Lanes

- `bit`
- `bus[W]`

### Event predicates

- `rise(sig)`
- `fall(sig)`
- `high(sig)`
- `low(sig)`
- `eq(sig, VALUE)`
- `neq(sig, VALUE)`
- `change(sig)`
- `stable(sig)`
- `stable({a, b, c})`

### Rule forms

- `not EVENT_A before EVENT_B`
- `EVENT_A -> after [MIN:MAX] EVENT_B`
- `PREDICATE_EXPR from EVENT_A until EVENT_B`

## Current Limitations

- Single clock domain only
- No capture/sample variables yet
- No multi-clock or async semantics
- Predicate expressions are conjunction-only in the current implementation
- Response rules render as routed top-band spans and hold rules render as lane highlights
- `not ... before ...` rules are summarized in the footer rather than drawn as a dedicated overlay

## Common Workflow

```bash
# 1. Validate the DSL
sva-timing validate examples/diagram_dsl/axi_wait.tdg

# 2. Render the timing diagram
sva-timing render examples/diagram_dsl/axi_wait.tdg --output out/axi_wait.svg

# 3. Emit parameterized SVA
sva-timing emit-sva examples/diagram_dsl/axi_wait.tdg --output out/axi_wait.sv
```

## Related Documents

- [Timing System Overview](architecture/timing-system-overview.md)
- [SVA Generator](sva-gen.md)
