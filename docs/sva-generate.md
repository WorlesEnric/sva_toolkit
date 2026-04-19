# SVA Generation

## Purpose

`sva_toolkit.generate` ports the V1 generator into the V3 package structure. It synthesizes legal SVA properties, supports random and stratified generation modes, computes construct coverage, and can optionally validate generated code with Verible.

## CLI Commands

Generate properties with the default random mode:

```bash
sva generate --count 3
```

Generate with stratified sampling and coverage output:

```bash
sva generate --count 3 --mode stratified --coverage
```

Validate the generated module with Verible when available:

```bash
sva generate --count 5 --validate
```

`--validate` requires `verible-verilog-syntax` on `PATH`.

## Usage Examples

Generate a module directly:

```python
from sva_toolkit.generate import SVASynthesizer

synth = SVASynthesizer(signals=["req", "ack", "gnt"], max_depth=2)
module_code, properties = synth.generate_module("demo_sva", 4)
print(module_code)
```

Use stratified generation:

```python
from sva_toolkit.generate import StratifiedGenerator

generator = StratifiedGenerator(signals=["req", "ack", "gnt"], samples_per_construct=2)
dataset = generator.generate_stratified_dataset()
```

Compute coverage:

```python
from sva_toolkit.generate import compute_coverage_statistics

stats = compute_coverage_statistics([prop.sva_code for prop in properties])
print(stats["coverage_pct"])
```

## API Reference

Primary public classes:

- `SVASynthesizer`
- `StratifiedGenerator`
- `SVAProperty`
- `GenerationResult`
- `ValidationResult`

Common helpers:

- `compute_coverage_statistics(properties: list[str]) -> dict[str, float]`
- `generate_sv_module(module_name, signals, property_bodies) -> str`
- `generate_assertion_only(property_body) -> str`
- `generate_cover_property(property_body) -> str`
- `generate_assume_property(property_body) -> str`

Signal presets:

- `DEFAULT_SIGNALS`
- `HANDSHAKE_SIGNALS`
- `FIFO_SIGNALS`
- `AXI_SIGNALS`

## V3 Notes

- The CLI intentionally keeps generation focused: count, mode, optional validation, and optional coverage.
- The Python API exposes the deeper controls such as signal selection, recursion depth, and direct template helpers.
- Verible is optional and only used for validation paths.

## Related Docs

- [Architecture](architecture.md)
- [Description engine](sva-describe.md)
- [Data workflows](sva-data.md)
