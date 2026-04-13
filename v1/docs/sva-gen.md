# SVA Generator (`sva-gen`)

## Overview

The SVA Generator is a type-directed synthesis tool that generates syntactically legal SystemVerilog Assertion properties through algorithmic composition. Unlike LLM-based generation, this tool employs a formal type system and grammar-based approach to construct assertions that are guaranteed to parse correctly, making it ideal for creating high-quality training datasets, stress-testing verification tools, and exploring the SVA property space systematically.

## Theoretical Foundation

### Type-Directed Synthesis

The generator implements a type-directed synthesis approach based on the following principles:

1. **Type System**: SVA constructs are classified into four fundamental types:
   - `EXPR`: Boolean expressions and signals
   - `BOOL`: Boolean-valued expressions with temporal context
   - `SEQUENCE`: Temporal sequences (events occurring over time)
   - `PROPERTY`: Assertion properties (top-level specifications)

2. **Production Rules**: Each type has associated production rules that specify valid constructions:
   - `EXPR → signal | signal OP signal`
   - `SEQUENCE → BOOL | SEQUENCE ##[delay] SEQUENCE`
   - `PROPERTY → SEQUENCE | SEQUENCE |-> PROPERTY`

3. **Context-Free Generation**: Starting from the `PROPERTY` type, the generator recursively applies production rules while respecting type constraints, ensuring all generated code is well-typed and syntactically valid.

### Advantages Over LLM-Based Generation

1. **Guaranteed syntactic correctness**: Type system prevents ill-formed constructions
2. **Controllable complexity**: Depth parameter bounds recursion, enabling complexity control
3. **No training data required**: Fully algorithmic, no machine learning dependency
4. **Reproducibility**: Seeded random generation produces identical outputs
5. **High throughput**: Generates thousands of properties per second
6. **Coverage**: Systematically explores syntactic patterns

## Architecture

The generation pipeline consists of:

1. **Type System Definition**: Enumeration of SVA types and their constructors
2. **Signal Management**: User-defined or preset signal lists (e.g., handshake protocols)
3. **Recursive Generation**: Depth-bounded random walk through the type grammar
4. **Template Instantiation**: Properties are wrapped in SystemVerilog modules with clocking
5. **Syntax Validation**: Generated code is optionally validated using Verible parser
6. **Output Formatting**: Results are formatted as compilable SystemVerilog or JSON

## Use Cases

1. **Training dataset generation**: Create large-scale synthetic datasets for LLM fine-tuning
2. **Fuzzing verification tools**: Stress-test formal verification engines with diverse properties
3. **Type system testing**: Validate that the type system prevents syntax errors (soundness testing)
4. **Benchmark construction**: Generate property suites with controlled complexity distributions
5. **Educational examples**: Produce diverse SVA examples for learning materials
6. **Property exploration**: Discover unusual but valid SVA patterns

## Command Reference

### Generate Assertions

Generate 10 SVA properties with default settings:

```bash
sva-gen generate -n 10
```

Output: SystemVerilog module printed to stdout with syntax highlighting.

### Generate with Custom Parameters

Control generation characteristics:

```bash
sva-gen generate \
  -n 50 \
  --max-depth 5 \
  --module-name my_assertions \
  --output assertions.sv \
  --seed 42
```

Parameters:
- `-n, --num-assertions`: Number of properties to generate (default: 10)
- `-d, --max-depth`: Maximum recursion depth (default: 4; higher = more complex)
- `-m, --module-name`: Generated module name (default: "sva_test_bench")
- `-o, --output`: Output file path (stdout if omitted)
- `--seed`: Random seed for reproducibility

### Use Signal Presets

Generate properties using predefined signal sets:

```bash
# AXI protocol signals
sva-gen generate -n 20 --preset axi

# Handshake protocol signals
sva-gen generate -n 20 --preset handshake

# FIFO control signals
sva-gen generate -n 20 --preset fifo
```

### Custom Signal List

Specify domain-specific signals:

```bash
sva-gen generate \
  -n 15 \
  -s packet_valid \
  -s packet_ready \
  -s header_complete \
  -s payload_valid \
  --clock pkt_clk
```

### JSON Output

Export properties as structured JSON:

```bash
sva-gen generate -n 10 --json-output > properties.json
```

JSON format:
```json
{
  "module_code": "module sva_test_bench;\n  logic clk;\n  ...",
  "properties": [
    "req |-> ##[1:2] ack",
    "valid && ready |=> data_en"
  ],
  "validation": {
    "is_valid": true,
    "error_message": null
  }
}
```

### Generate Without Validation

Skip Verible validation (faster generation):

```bash
sva-gen generate -n 100 --no-validate
```

### Validate Existing SVA

Check syntax of an SVA property:

```bash
sva-gen validate "assert property (@(posedge clk) req |-> ##1 ack);"
```

### Validate SystemVerilog File

Verify syntax of a complete file:

```bash
sva-gen validate-file assertions.sv
```

### List Available Presets

Display predefined signal sets:

```bash
sva-gen list-presets
```

Output:
```
Signal Presets
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Preset    ┃ Signals                              ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ default   │ req, ack, gnt, valid, ready, ...     │
│ handshake │ req, ack, valid, ready, grant, ...   │
│ fifo      │ push, pop, full, empty, ...          │
│ axi       │ awvalid, awready, wvalid, ...        │
└───────────┴──────────────────────────────────────┘
```

### Stress Test

Test the type system for soundness (verify no syntax errors leak through):

```bash
sva-gen stress-test -n 100 -s 50 --max-depth 6
```

Parameters:
- `-n, --num-batches`: Number of batches
- `-s, --batch-size`: Properties per batch
- `-d, --max-depth`: Recursion depth

Output:
```
SVA Generator Stress Test
  Batches: 100, Batch Size: 50
  Total Assertions: 5000

Stress Test Results
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric            ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Assertions  │ 5000  │
│ Valid Batches     │ 100   │
│ Invalid Batches   │ 0     │
│ Success Rate      │ 100%  │
└───────────────────┴───────┘
```

## Signal Presets

The tool includes predefined signal sets for common hardware protocols:

### Default Preset
```systemverilog
req, ack, gnt, valid, ready, data_en, busy
```

### Handshake Preset
```systemverilog
req, ack, valid, ready, grant, request
```

### FIFO Preset
```systemverilog
push, pop, full, empty, almost_full, almost_empty
```

### AXI Preset
```systemverilog
awvalid, awready, wvalid, wready, bvalid, bready,
arvalid, arready, rvalid, rready
```

## Integration Examples

### Large-Scale Dataset Generation

```python
# generate_training_data.py
import subprocess
import json

def generate_sva_dataset(num_samples, output_file, seed_start=0):
    """Generate synthetic SVA dataset for training."""
    dataset = []

    for i in range(num_samples):
        result = subprocess.run(
            ["sva-gen", "generate",
             "-n", "1",
             "--max-depth", str(3 + (i % 4)),  # Vary complexity
             "--seed", str(seed_start + i),
             "--json-output"],
            capture_output=True,
            text=True
        )

        data = json.loads(result.stdout)
        sva_code = data["properties"][0]

        dataset.append({
            "id": f"synthetic_{i:06d}",
            "SVA": sva_code,
            "source": "type_directed_synthesis",
            "depth": 3 + (i % 4),
            "seed": seed_start + i
        })

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {num_samples} properties → {output_file}")

# Generate 10,000 synthetic properties
generate_sva_dataset(10000, "synthetic_dataset.json")
```

### Complexity-Stratified Generation

```python
# stratified_generation.py
import subprocess
import json

def generate_by_depth(depth, count, preset="default"):
    """Generate properties of specific complexity."""
    result = subprocess.run(
        ["sva-gen", "generate",
         "-n", str(count),
         "--max-depth", str(depth),
         "--preset", preset,
         "--json-output"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)["properties"]

# Generate stratified dataset
dataset = {
    "simple": generate_by_depth(2, 100),      # Shallow properties
    "medium": generate_by_depth(4, 100),      # Medium complexity
    "complex": generate_by_depth(6, 100),     # Deep nesting
}

# Save by category
for category, properties in dataset.items():
    with open(f"properties_{category}.json", "w") as f:
        json.dump(properties, f, indent=2)
```

### Verification Tool Fuzzing

```bash
#!/bin/bash
# fuzz_verification_tool.sh - Stress-test formal verification engine

TOOL="ebmc"
OUTPUT_DIR="fuzz_results"

mkdir -p "$OUTPUT_DIR"

for i in {1..1000}; do
    echo "Fuzzing iteration $i"

    # Generate random property
    sva-gen generate -n 1 --seed $i --output "$OUTPUT_DIR/test_$i.sv"

    # Attempt verification
    timeout 30s $TOOL "$OUTPUT_DIR/test_$i.sv" \
        > "$OUTPUT_DIR/test_$i.log" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "CRASH: Iteration $i (exit code: $EXIT_CODE)"
        cp "$OUTPUT_DIR/test_$i.sv" "$OUTPUT_DIR/crash_$i.sv"
    fi
done

echo "Fuzzing complete. Check $OUTPUT_DIR for crashes."
```

### Reproducible Generation Pipeline

```python
# reproducible_generation.py
import subprocess
import hashlib

def generate_reproducible(seed, depth, num_props):
    """Generate properties with guaranteed reproducibility."""
    result = subprocess.run(
        ["sva-gen", "generate",
         "-n", str(num_props),
         "--max-depth", str(depth),
         "--seed", str(seed),
         "--json-output"],
        capture_output=True,
        text=True
    )

    data = json.loads(result.stdout)
    properties = data["properties"]

    # Compute checksum for verification
    content = "\n".join(properties)
    checksum = hashlib.sha256(content.encode()).hexdigest()

    return {
        "properties": properties,
        "seed": seed,
        "depth": depth,
        "checksum": checksum
    }

# Generate and verify
config = {"seed": 42, "depth": 4, "num_props": 100}

run1 = generate_reproducible(**config)
run2 = generate_reproducible(**config)

assert run1["checksum"] == run2["checksum"], "Generation not reproducible!"
print(f"✓ Reproducibility verified (checksum: {run1['checksum'][:16]}...)")
```

## Type System Details

### Type Hierarchy

```
SVANode (abstract base)
├── Signal (type: EXPR)
├── BinaryOp (type: BOOL)
│   └── operators: &&, ||, ^, ==, !=
├── UnarySysFunction (type: BOOL)
│   └── functions: $rose, $fell, $stable, $past
├── SequenceDelay (type: SEQUENCE)
│   └── ##[min:max]
├── SequenceRepeat (type: SEQUENCE)
│   └── [*min:max]
├── SequenceBinary (type: SEQUENCE)
│   └── operators: and, or, intersect
├── Implication (type: PROPERTY)
│   └── |-> (overlapped), |=> (non-overlapped)
├── DisableIff (type: PROPERTY)
└── NotProperty (type: PROPERTY)
```

### Production Rules

The generator selects production rules based on current type and desired output type:

```python
TYPE_EXPR → Signal
         | BinaryOp(TYPE_BOOL, TYPE_BOOL)

TYPE_BOOL → TYPE_EXPR
         | UnarySysFunction(TYPE_EXPR)
         | BinaryOp(TYPE_BOOL, TYPE_BOOL)

TYPE_SEQUENCE → TYPE_BOOL
              | SequenceDelay(TYPE_SEQUENCE)
              | SequenceRepeat(TYPE_SEQUENCE)
              | SequenceBinary(TYPE_SEQUENCE, TYPE_SEQUENCE)

TYPE_PROPERTY → TYPE_SEQUENCE
              | Implication(TYPE_SEQUENCE, TYPE_PROPERTY)
              | DisableIff(TYPE_BOOL, TYPE_PROPERTY)
              | NotProperty(TYPE_PROPERTY)
```

## Performance Characteristics

### Generation Speed

On a modern CPU (Apple M1/Intel i7):
- Simple properties (depth 2-3): ~10,000 properties/second
- Medium properties (depth 4-5): ~5,000 properties/second
- Complex properties (depth 6-8): ~1,000 properties/second

With Verible validation:
- Validation overhead: ~50-100ms per property
- Effective throughput: ~10-20 properties/second

### Memory Usage

- Per-property memory: ~1-10 KB (depends on depth)
- Module template: ~5 KB overhead
- Batch generation (1000 properties): ~10-50 MB total

## Limitations

1. **Semantic meaningfulness**: Generated properties are syntactically correct but may lack semantic coherence
2. **No design context**: Does not consider actual hardware behavior or realistic property patterns
3. **Limited temporal complexity**: Very deep temporal nesting (>10 levels) may exceed practical bounds
4. **Validation dependency**: Syntax checking requires Verible installation
5. **Randomness**: Without seeding, output is non-deterministic

## Soundness Guarantee

The type system provides a **soundness guarantee**:

**Theorem**: All properties generated by the type-directed synthesis algorithm are syntactically valid SystemVerilog Assertions (assuming correct type system implementation).

**Empirical Validation**: The `stress-test` command validates this claim by generating thousands of properties and verifying 100% parse successfully.

## Best Practices

1. **Use seeding for reproducibility**: Always set `--seed` for scientific experiments
2. **Control complexity**: Adjust `--max-depth` based on use case (2-4 for simple, 5-7 for complex)
3. **Validate periodically**: Run `stress-test` after modifying type system
4. **Domain-specific signals**: Use custom signal lists for targeted domain coverage
5. **Batch generation**: Generate in large batches for efficiency
6. **Version control**: Track tool version and configuration in dataset metadata

## Troubleshooting

### Verible Not Found

```
Error: Verible binary not found
```

Solution:
```bash
# Download Verible from GitHub releases
wget https://github.com/chipsalliance/verible/releases/download/v0.0-3000/verible-v0.0-3000-Ubuntu-20.04-focal-x86_64.tar.gz

# Extract and set path
sva-gen --verible-path /path/to/verible-verilog-syntax generate ...
```

### Type System Soundness Violation

If `stress-test` reports failures:

```
Error: Syntax error detected (type system leak)
```

This indicates a bug in the type system implementation. Please report with:
1. Full command used
2. Random seed that triggers the error
3. Generated property that fails validation

### Generation Too Slow

For faster generation:

```bash
# Disable validation
sva-gen generate -n 10000 --no-validate
```

## Future Enhancements

Potential extensions to the generator:

1. **Semantic constraints**: Add domain-specific rules (e.g., "req before ack")
2. **Coverage-guided generation**: Target specific SVA constructs systematically
3. **Mutation-based generation**: Evolve existing properties through mutations
4. **Template-based generation**: Combine type-directed synthesis with parametric templates
5. **Multi-clock domains**: Support properties spanning multiple clock domains

## References

- SystemVerilog IEEE 1800-2017 Standard, Section 16: Assertions
- Type-Directed Program Synthesis (Osera & Zdancewic, 2015)
- Verible SystemVerilog Parser: https://github.com/chipsalliance/verible
- Grammar-Based Fuzzing for Language Processors (survey paper)
