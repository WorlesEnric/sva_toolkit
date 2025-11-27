# SVA Toolkit

A comprehensive toolkit for SystemVerilog Assertion (SVA) generation, validation, and LLM training.

## Features

- **SVA AST Parser**: Parse and extract structured information from SVA code using Verible
- **SVA Implication Checker**: Verify implication relationships between SVA pairs using SymbiYosys
- **SVA CoT Builder**: Generate Chain-of-Thought reasoning from SVA AST
- **Dataset Builder**: Build training datasets with SVAD and CoT annotations
- **Benchmark**: Evaluate LLM performance on SVA generation tasks

## Installation

```bash
pip install -e .
```

### Prerequisites

- [Verible](https://github.com/chipsalliance/verible) - for SVA parsing
- [SymbiYosys](https://github.com/YosysHQ/sby) - for formal verification
- [Yosys](https://github.com/YosysHQ/yosys) - synthesis tool

## CLI Usage

### SVA AST Parser

```bash
# Parse SVA and extract structure
sva-ast parse "property p; a |-> b; endproperty"

# Parse from file
sva-ast parse-file input.sv
```

### SVA Implication Checker

```bash
# Check if SVA1 implies SVA2
sva-implication check --antecedent "a |-> b" --consequent "a |-> ##1 b"

# Check equivalence (bidirectional implication)
sva-implication equivalent --sva1 "a |-> b" --sva2 "!a || b"
```

### SVA CoT Builder

```bash
# Generate CoT from SVA
sva-cot build "property p; @(posedge clk) req |-> ##[1:3] ack; endproperty"
```

### Dataset Builder

```bash
# Build dataset with SVAD and CoT
sva-dataset build input.json output.json \
    --base-url "https://api.openai.com/v1" \
    --model "gpt-4" \
    --api-key "your-api-key"
```

### Benchmark

```bash
# Run benchmark on dataset
sva-benchmark run dataset.json \
    --base-url "https://api.openai.com/v1" \
    --model "gpt-4" \
    --api-key "your-api-key"
```

## Project Structure

```
sva_toolkit/
├── src/sva_toolkit/
│   ├── ast_parser/      # Verible-based SVA parsing
│   ├── implication_checker/  # SymbiYosys-based verification
│   ├── cot_builder/     # Chain-of-Thought generation
│   ├── dataset_builder/ # Dataset construction
│   ├── benchmark/       # LLM evaluation
│   └── utils/           # Shared utilities
├── templates/           # Jinja2 templates
└── tests/              # Unit tests
```

## License

MIT License
