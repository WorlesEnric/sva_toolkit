# Timing Diagrams

## Purpose

`sva_toolkit.timing` owns the timing-diagram DSL, rendering pipeline, and bridges to and from SVA. In V3 it remains the canonical way to move between diagram-oriented protocol documentation and assertion-oriented verification artifacts.

## CLI Commands

Validate timing DSL:

```bash
sva timing validate examples/td/01_simple_handshake.td
```

Render SVG:

```bash
sva timing render examples/td/01_simple_handshake.td -o examples/out/01_simple_handshake.svg
```

Render PNG:

```bash
sva timing render examples/td/01_simple_handshake.td --format png -o examples/out/01_simple_handshake.png
```

Emit SVA from timing DSL:

```bash
sva timing emit-sva examples/td/11_emit_sva_bridge.td -o examples/out/11_emit_sva_bridge.sv
```

Extract timing DSL from SVA:

```bash
sva timing extract-sva examples/sva/11_emit_sva_bridge.sv -o examples/out/11_emit_sva_bridge.td
```

Bundle related SVA files:

```bash
sva timing bundle-sva examples/sva/11_emit_sva_bridge.sv examples/sva/12_extract_sva_bridge.sv -o examples/out/bundled.td
```

## Usage Examples

Parse a diagram:

```python
from pathlib import Path

from sva_toolkit.timing.frontend.parser import parse_diagram

document = parse_diagram(Path("examples/td/01_simple_handshake.td").read_text())
print(document.name, document.ticks)
```

Render SVG from Python:

```python
from pathlib import Path

from sva_toolkit.timing import render_diagram_svg
from sva_toolkit.timing.frontend.parser import parse_diagram

document = parse_diagram(Path("examples/td/01_simple_handshake.td").read_text())
svg = render_diagram_svg(document)
```

Emit parameterized SVA:

```python
from pathlib import Path

from sva_toolkit.timing.bridge.emit_sva import emit_parameterized_sva
from sva_toolkit.timing.frontend.parser import parse_diagram

document = parse_diagram(Path("examples/td/11_emit_sva_bridge.td").read_text())
print(emit_parameterized_sva(document))
```

## API Reference

Stable public exports:

- `render_diagram_svg(document) -> str`
- `render_diagram_png(document, path) -> None`

Frequently used internal modules:

- `sva_toolkit.timing.frontend.parser.parse_diagram`
- `sva_toolkit.timing.frontend.validate.validate_diagram`
- `sva_toolkit.timing.bridge.emit_sva.emit_parameterized_sva`
- `sva_toolkit.timing.bridge.from_sva.extract_sva_scenario`
- `sva_toolkit.timing.bridge.from_sva.bundle_sva_scenarios`
- `sva_toolkit.timing.bridge.to_dsl.emit_timing_dsl`

Core model classes live under `sva_toolkit.timing.core.scenario`, including:

- `ScenarioDocument`
- `ClockingSpec`
- `SignalDecl`
- `Anchor`
- `TimeWindow`
- `PropertyOverlay`

## V3 Notes

- SVG rendering is available in the base install.
- PNG rendering requires the `timing-render` extra because it uses `cairosvg`.
- The `examples/td/` and `examples/sva/` directories in V3 are ported from the V2 timing example suite.
- Extraction and bundling are best-effort structure recovery workflows, not full semantic proof steps.

## Related Docs

- [Architecture](architecture.md)
- [Formal verification](sva-formal.md)
- [Timing diagram dataset generation](timing-dataset-generation.md)
- [Examples](../examples/README.md)
