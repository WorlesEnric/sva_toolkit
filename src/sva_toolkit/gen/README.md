# SVA-Gen: Type-Directed SystemVerilog Assertion Generator

SVA-Gen is a programmatic engine designed to generate syntactically legal and "rich" SystemVerilog Assertions (SVA). Unlike simple string-based generators, SVA-Gen uses a Type-Directed Synthesis approach to ensure that temporal logic, boolean expressions, and property implications are always correctly nested.

## Installation

The generator is part of the sva_toolkit package:

```bash
pip install -e .
```

## CLI Usage

### Generate SVA Properties

```bash
# Basic generation (10 properties with validation)
sva-gen generate

# Generate 20 properties with custom settings
sva-gen generate -n 20 -d 5 -m my_module -o output.sv

# Use specific signals
sva-gen generate -s req -s ack -s valid -s ready

# Use predefined signal presets
sva-gen generate --preset axi

# Disable validation
sva-gen generate --no-validate

# Output as JSON
sva-gen generate -j -o output.json

# Reproducible generation with seed
sva-gen generate --seed 42
```

### Validate SVA Syntax

```bash
# Validate a file
sva-gen validate-file generated.sv

# Validate inline code
sva-gen validate "property p; @(posedge clk) a |-> b; endproperty"
```

### Stress Test

Run stress test to find type system leaks:

```bash
# Run 100 batches of 10 assertions each
sva-gen stress-test -n 100 -s 10
```

### List Signal Presets

```bash
sva-gen list-presets
```

## Programmatic Usage

```python
from sva_toolkit.gen import SVASynthesizer

# Create synthesizer
synthesizer = SVASynthesizer(
    signals=["req", "ack", "gnt", "valid"],
    max_depth=4,
    clock_signal="clk"
)

# Generate a single property
prop = synthesizer.synthesize(name="p_handshake")
print(prop)

# Generate a complete module with validation
result = synthesizer.generate_validated(
    module_name="my_assertions",
    num_assertions=10
)

if result.validation.is_valid:
    print("All assertions are syntactically valid!")
    print(result.module_code)
else:
    print(f"Validation error: {result.validation.error_message}")
```

## Project Structure

- `types_sva.py`: The core Type System. Defines classes for Boolean, Sequence, and Property.
- `generator.py`: The recursive synthesis engine that builds expression trees based on type rules.
- `templates.py`: SystemVerilog boilerplate to wrap properties into a module.
- `utils.py`: Randomization helpers and weighting logic.
- `cli.py`: The Click-based CLI entry point.

## How It Works

The generator follows the SVA promotion hierarchy:

```
Expressions (Signals, Arithmetic) ->
Booleans (Comparisons, System Functions) ->
Sequences (Temporal delays ##, repetitions [*], binary ops intersect) ->
Properties (Implications |->, disable iff, not).
```

## Syntax Validation

Generated SVAs are validated using Verible. The validation works by:

1. Wrapping generated properties in a dummy module
2. Running `verible-verilog-syntax` on the module
3. Checking the return code for syntax errors

If validation fails, it indicates a "type system leak" - a rule that allows invalid syntax.

## Customization

To add new SVA keywords (e.g., `matched`, `first_match`):

1. Add a new class in `types_sva.py` inheriting from `SVANode`
2. Update the corresponding `generate_*` method in `generator.py` to include the new class in the production rules

## Signal Presets

Available presets:
- `default`: req, ack, gnt, valid, ready, data_en, busy
- `handshake`: req, ack, valid, ready, grant, request
- `fifo`: push, pop, full, empty, almost_full, almost_empty
- `axi`: awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready
