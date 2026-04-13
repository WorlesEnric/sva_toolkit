# SVA AST Parser (`sva-ast`)

## Overview

The SVA AST (Abstract Syntax Tree) Parser is a specialized static analysis tool designed to extract structured syntactic and semantic information from SystemVerilog Assertion (SVA) code. The parser leverages the Verible syntax analysis engine to construct an intermediate representation that captures essential components of SVA properties, including temporal operators, signal dependencies, implication structures, and timing constraints.

## Technical Background

SystemVerilog Assertions provide a declarative framework for specifying temporal and functional properties of digital hardware designs. Understanding the structure of these assertions is critical for automated verification workflows, formal property synthesis, and machine learning applications targeting hardware verification. The AST parser bridges the gap between raw SVA source code and machine-readable structured representations suitable for downstream analysis.

The parser extracts multiple levels of information:

1. **Structural components**: Property names, assertion types (assert, assume, cover), and overall property classification
2. **Clocking information**: Clock signals, edge specifications (posedge/negedge), and synchronous timing domains
3. **Reset handling**: Reset signal identification, polarity detection (active-high/active-low), and disable conditions
4. **Implication relationships**: Antecedent-consequent structure, implication operators (|-> overlapped, |=> non-overlapped)
5. **Signal analysis**: Complete signal enumeration with role classification (clock, reset, data)
6. **Temporal semantics**: Delay operators, sequence repetition, and built-in temporal functions
7. **Built-in functions**: System functions such as $rose, $fell, $stable, and $past

## Architecture

The parser operates in two phases:

1. **Lexical and syntactic analysis**: Verible-based parsing generates a concrete syntax tree from the input SVA code
2. **Semantic extraction**: Custom traversal algorithms extract domain-specific SVA constructs into a normalized representation

The output is serializable to JSON format, enabling integration with Python-based machine learning pipelines, formal verification toolchains, and automated property generation systems.

## Use Cases

1. **Property comprehension**: Automated extraction of property semantics for documentation generation
2. **Dataset construction**: Preprocessing SVA code for supervised learning applications
3. **Signal dependency analysis**: Identification of all signals referenced within complex temporal properties
4. **Formal verification preprocessing**: Extraction of structural patterns for equivalence checking and implication analysis
5. **Property transformation**: Basis for automated refactoring and optimization of assertion suites

## Command Reference

### Parse SVA Code String

Parse an SVA property provided as a command-line argument:

```bash
sva-ast parse "assert property (@(posedge clk) disable iff (rst) req |-> ##[1:3] ack);"
```

Output format: Rich-formatted tables displaying property structure, signals, temporal operators, and delays.

### Parse SVA Code with JSON Output

Generate machine-readable JSON representation:

```bash
sva-ast parse "property p1; @(posedge clk) (valid && ready) |=> data_out; endproperty" --json-output
```

### Parse SVA from File

Process SVA properties contained in a SystemVerilog file:

```bash
sva-ast parse-file assertions.sv
```

This command extracts all assertion properties from the file and displays each structure sequentially.

### Parse File with JSON Output

Export all properties from a file to JSON format:

```bash
sva-ast parse-file assertions.sv --json-output > output.json
```

### Extract Signals from SVA

List all signals referenced within an SVA property:

```bash
sva-ast signals "assert property (@(posedge clk) req |-> ##1 gnt);"
```

Output: Alphabetically sorted table of signal names with role annotations (clock, reset, data).

### Custom Verible Path

Specify a custom path to the Verible syntax checker binary:

```bash
sva-ast --verible-path /custom/path/verible-verilog-syntax parse "property p; ... endproperty"
```

## Output Schema

The JSON output schema includes the following fields:

```json
{
  "property_name": "string | null",
  "is_assertion": "boolean",
  "is_assumption": "boolean",
  "is_cover": "boolean",
  "clock_signal": "string | null",
  "clock_edge": "posedge | negedge | null",
  "reset_signal": "string | null",
  "reset_active_low": "boolean",
  "disable_condition": "string | null",
  "implication_type": "overlapped | non_overlapped | none",
  "antecedent": "string | null",
  "consequent": "string | null",
  "signals": [
    {
      "name": "string",
      "is_clock": "boolean",
      "is_reset": "boolean"
    }
  ],
  "builtin_functions": [
    {
      "name": "string",
      "arguments": ["string"]
    }
  ],
  "temporal_operators": ["string"],
  "delays": [
    {
      "min_cycles": "integer",
      "max_cycles": "integer | null",
      "is_unbounded": "boolean"
    }
  ]
}
```

## Integration Examples

### Python Integration

```python
import subprocess
import json

result = subprocess.run(
    ["sva-ast", "parse", "--json-output", sva_code],
    capture_output=True,
    text=True
)
structure = json.loads(result.stdout)
print(f"Signals: {[s['name'] for s in structure['signals']]}")
```

### Batch Processing

```bash
for file in tests/*.sv; do
    echo "Processing $file"
    sva-ast parse-file "$file" --json-output >> dataset.jsonl
done
```

## Limitations

1. **Syntax validation dependency**: Requires valid SystemVerilog syntax accepted by Verible
2. **Static analysis only**: Does not perform semantic validation or formal verification
3. **Limited coverage**: Focuses on property-level constructs; does not parse full module hierarchies
4. **Verible dependency**: Parser accuracy depends on Verible version and capabilities

## References

- SystemVerilog IEEE 1800-2017 Standard, Section 16: Assertions
- Verible SystemVerilog Parser: https://github.com/chipsalliance/verible
