# SVA Toolkit Documentation

This directory contains comprehensive academic documentation for all tools in the SVA Toolkit. Each document provides theoretical background, architectural details, use cases, and extensive command examples.

## Available Tools

### 1. [SVA AST Parser (`sva-ast`)](sva-ast.md)
**Purpose**: Extract structured syntactic and semantic information from SystemVerilog Assertion code.

**Key Features**:
- Parse SVA properties into machine-readable JSON format
- Extract signals, temporal operators, delays, and implication structures
- Support for both inline code and file-based parsing
- Verible-based syntax analysis

**Primary Use Case**: Preprocessing SVA code for machine learning datasets and formal analysis.

---

### 2. [SVA Implication Checker (`sva-implication`)](sva-implication.md)
**Purpose**: Verify logical implication relationships between SVA pairs using bounded model checking via EBMC.

**Key Features**:
- Check if one property implies another (P → Q)
- Verify bidirectional equivalence (P ⇔ Q)
- Batch processing of property pairs
- EBMC-based formal verification

**Primary Use Case**: Validating LLM-generated assertions against reference specifications.

---

### 3. [SVA VCFormal Implication Checker (`sva-vcformal-implication`)](sva-vcformal-implication.md)
**Purpose**: Implication checking using Cadence VC Formal with cross-validation capabilities.

**Key Features**:
- Unbounded formal proofs via VC Formal
- Cross-validation against EBMC results
- Advanced SAT-based verification algorithms
- Batch processing with discrepancy detection

**Primary Use Case**: High-assurance verification and tool comparison studies.

---

### 4. [SVA Chain-of-Thought Builder (`sva-cot`)](sva-cot.md)
**Purpose**: Generate structured natural language explanations of SVA semantics.

**Key Features**:
- Template-based reasoning trace generation
- Markdown-formatted output
- Section-based decomposition (overview, temporal analysis, etc.)
- Integration with documentation systems

**Primary Use Case**: Creating training data for LLMs and automated property documentation.

---

### 5. [SVA Dataset Builder (`sva-dataset`)](sva-dataset.md)
**Purpose**: Construct high-quality training datasets with SVAD and CoT annotations.

**Key Features**:
- LLM-based SVAD (description) generation
- Template-based CoT generation
- Multiprocessing support for scalability
- Persistent caching for fault tolerance
- Custom system prompt support

**Primary Use Case**: Building datasets for fine-tuning code generation models.

---

### 6. [SVA Benchmark Runner (`sva-benchmark`)](sva-benchmark.md)
**Purpose**: Evaluate LLM performance on SVA generation tasks with formal semantic verification.

**Key Features**:
- End-to-end benchmarking (generation + verification)
- Multi-model comparison with statistical metrics
- Equivalence rate, implication rate, and success rate tracking
- Multiprocessing and caching support
- Detailed per-item results and error analysis

**Primary Use Case**: Comparing LLMs for assertion generation quality.

---

### 7. [SVA Generator (`sva-gen`)](sva-gen.md)
**Purpose**: Generate syntactically valid SVA properties through type-directed synthesis.

**Key Features**:
- Guaranteed syntactic correctness via type system
- Configurable complexity (depth control)
- Signal presets for common protocols (AXI, FIFO, handshake)
- Reproducible generation with random seeds
- Stress testing mode for type system validation

**Primary Use Case**: Creating synthetic training datasets and fuzzing verification tools.

---

## Documentation Standards

All documentation follows these conventions:

1. **Academic tone**: Formal, precise language suitable for research papers
2. **Theoretical background**: Conceptual foundations and motivation
3. **Architecture**: High-level design and component interaction
4. **Use cases**: Practical application scenarios
5. **Command reference**: Comprehensive examples with expected outputs
6. **Integration examples**: Code snippets for Python/Bash automation
7. **Limitations**: Explicit constraints and known issues

## Quick Start Guide

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/sva_toolkit.git
cd sva_toolkit

# Install in editable mode
pip install -e .

# Verify installation
sva-gen --help
sva-benchmark --help
```

### Common Workflows

#### 1. Generate Synthetic Dataset
```bash
# Generate 1000 properties
sva-gen generate -n 1000 --json-output > synthetic.json

# Add CoT annotations
sva-dataset build-cot-only synthetic.json annotated.json
```

#### 2. Build Training Dataset from SVA Files
```bash
# Collect SVA files
find ./rtl -name "*.sv" -exec cat {} \; > all_sva.txt

# Build dataset with SVAD + CoT
sva-dataset build raw_sva.json final_dataset.json \
  --base-url https://api.openai.com/v1 \
  --model gpt-4 \
  --api-key $OPENAI_API_KEY
```

#### 3. Benchmark LLM Models
```bash
# Prepare config file
cat > models.json << 'EOF'
[
  {"base_url": "...", "model": "gpt-4o", "api_key": "..."},
  {"base_url": "...", "model": "claude-3-opus", "api_key": "..."}
]
EOF

# Run benchmark
sva-benchmark run-multi benchmark_dataset.json \
  --config-file models.json \
  --output results.json
```

#### 4. Validate Generated Assertions
```bash
# Check syntax
sva-gen validate-file generated_assertions.sv

# Check semantic equivalence
sva-implication equivalent \
  --sva1 "reference_property" \
  --sva2 "generated_property"
```

## Tool Integration Matrix

| Tool | Inputs | Outputs | Dependencies |
|------|--------|---------|--------------|
| `sva-ast` | SVA code | JSON structure | Verible |
| `sva-implication` | 2 SVA properties | Implication result | EBMC |
| `sva-vcformal-implication` | 2 SVA properties | Implication result | VC Formal, EBMC (optional) |
| `sva-cot` | SVA code | Markdown explanation | None |
| `sva-dataset` | SVA list | Annotated dataset | LLM API (optional) |
| `sva-benchmark` | (SVAD, SVA) pairs | Performance metrics | LLM API, EBMC, Verible |
| `sva-gen` | Signal list | SVA properties | Verible (optional) |

## Citation

If you use this toolkit in academic work, please cite:

```bibtex
@software{sva_toolkit,
  title = {SVA Toolkit: A Comprehensive Suite for SystemVerilog Assertion Generation and Verification},
  author = {Qihang Wang},
  year = {2025},
}
```

## Support and Contribution

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join the community forum for questions
- **Contributing**: See CONTRIBUTING.md for development guidelines

## License

MIT License - See LICENSE file for details.
