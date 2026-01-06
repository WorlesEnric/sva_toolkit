# SVA VCFormal Implication Checker (`sva-vcformal-implication`)

## Overview

The VCFormal-based Implication Checker is an industrial-grade formal verification tool that leverages Cadence VC Formal (VCF) as its primary verification engine for determining implication relationships between SystemVerilog Assertion pairs. This tool provides an alternative to bounded model checking approaches, utilizing advanced formal techniques including unbounded proof methods, SAT-based engines, and semi-formal simulation-guided verification.

## Rationale for VCFormal Integration

While bounded model checkers (e.g., EBMC) provide sound verification within a fixed depth bound, commercial formal tools like VC Formal offer several advantages:

1. **Unbounded proofs**: VCFormal can construct inductive proofs that hold for all time steps, not just bounded depths
2. **Advanced algorithms**: Incorporates state-of-the-art SAT solvers, interpolation, and property decomposition techniques
3. **Industrial verification**: Proven on large-scale commercial designs with complex temporal properties
4. **Counterexample quality**: Provides minimal, human-readable counterexamples with waveform generation
5. **Performance optimization**: Adaptive engine selection based on property characteristics

This tool is designed for scenarios where high-confidence verification is required, particularly when validating critical properties or when bounded verification yields inconclusive results.

## Architecture

The verification flow incorporates the following stages:

1. **Property instrumentation**: SVA properties are embedded into a synthesizable SystemVerilog module with appropriate clocking and reset structures
2. **VCFormal invocation**: The VCF engine is invoked via command-line interface with optimized proof strategies
3. **Result parsing**: Tool output is analyzed to extract proof results (PROVEN, FALSIFIED, INCONCLUSIVE)
4. **Cross-validation (optional)**: Results can be cross-checked against EBMC to identify potential soundness issues or engine-specific behaviors
5. **Report generation**: Structured output includes verification status, execution time, and diagnostic information

## Cross-Validation Capability

A unique feature of this tool is its ability to cross-validate VCFormal results against EBMC, enabling:

1. **Soundness checking**: Detecting discrepancies between different verification engines
2. **Confidence boosting**: Agreement between tools increases confidence in results
3. **Engine characterization**: Understanding which types of properties each engine handles effectively
4. **Research applications**: Empirical studies on formal verification tool reliability

## Use Cases

1. **High-assurance verification**: Critical properties requiring unbounded proofs
2. **Tool comparison studies**: Benchmarking VCFormal against open-source alternatives
3. **Industrial workflows**: Integration with Cadence verification environments
4. **Assertion library validation**: Checking consistency of large property suites
5. **Automated property generation**: Validating machine-generated assertions against specifications

## Command Reference

### Check Implication

Verify whether antecedent implies consequent using VCFormal:

```bash
sva-vcformal-implication check \
  --antecedent "req && valid |-> ##[1:3] ack" \
  --consequent "req |-> ##1 ack"
```

### Check with Verbose Logging

Include detailed VCFormal execution logs:

```bash
sva-vcformal-implication --verbose check \
  --antecedent "a ##1 b" \
  --consequent "a ##2 b" \
  --show-log
```

### Check Equivalence

Determine if two SVAs are semantically equivalent:

```bash
sva-vcformal-implication equivalent \
  --sva1 "$rose(clk) |-> data_valid" \
  --sva2 "(!clk ##1 clk) |-> data_valid"
```

### Determine Full Relationship

Analyze bidirectional implication:

```bash
sva-vcformal-implication relationship \
  --sva1 "req |-> ##[1:2] ack" \
  --sva2 "req |-> ##1 ack"
```

Output interpretation:
- `sva1_implies_sva2: true` means SVA1 is stronger (more restrictive)
- `sva2_implies_sva1: true` means SVA2 is stronger
- Both true indicates equivalence
- Both false indicates no relationship

### Cross-Validation with EBMC

Run VCFormal and EBMC on a batch of property pairs, comparing results:

```bash
sva-vcformal-implication cross-validate \
  --input-file property_pairs.json \
  --output-file cross_validation_report.json \
  --ebmc-path /usr/local/bin/ebmc \
  --ebmc-depth 25 \
  --ebmc-timeout 300
```

Input JSON format:
```json
[
  {
    "id": "test_001",
    "antecedent": "req |-> ##1 gnt",
    "consequent": "req |-> ##[1:2] gnt"
  },
  {
    "id": "test_002",
    "sva1": "valid && ready",
    "sva2": "valid ##0 ready"
  }
]
```

Note: The tool accepts both `antecedent`/`consequent` and `sva1`/`sva2` field names.

### JSON Output

Export results in machine-readable format:

```bash
sva-vcformal-implication check \
  --antecedent "signal_a" \
  --consequent "signal_a && signal_b" \
  --json-output
```

### Custom VCFormal Path and Timeout

Configure tool paths and execution limits:

```bash
sva-vcformal-implication \
  --vcf-path /opt/cadence/vcf/bin/vcf \
  --timeout 600 \
  --work-dir ./vcf_work \
  --keep-files \
  check --antecedent "..." --consequent "..."
```

Configuration parameters:
- `--vcf-path`: Path to VC Formal executable
- `--timeout`: Maximum execution time per verification task (seconds)
- `--work-dir`: Directory for intermediate files
- `--keep-files`: Retain generated files for debugging

## Cross-Validation Workflow

### Batch Cross-Validation Example

```bash
# Prepare input file with 100 property pairs
cat > pairs.json << 'EOF'
[
  {"id": "pair_1", "sva1": "a |-> b", "sva2": "a |=> b"},
  {"id": "pair_2", "sva1": "req ##1 ack", "sva2": "req ##[1:1] ack"}
]
EOF

# Run cross-validation with both engines
sva-vcformal-implication -v cross-validate \
  -i pairs.json \
  -o results.json \
  --ebmc-path /usr/bin/ebmc \
  --ebmc-depth 20 \
  --max-mismatches 10
```

### Analyzing Cross-Validation Output

The output JSON contains:

```json
{
  "summary": {
    "total": 100,
    "aligned": 95,
    "mismatched": 3,
    "ebmc_skipped": 2,
    "vcformal_counts": {"IMPLIES": 80, "NOT_IMPLIES": 15, "ERROR": 5},
    "ebmc_counts": {"IMPLIES": 78, "NOT_IMPLIES": 17}
  },
  "mismatches": [
    {
      "id": "pair_42",
      "vcformal_result": "IMPLIES",
      "ebmc_result": "NOT_IMPLIES",
      "antecedent": "...",
      "consequent": "..."
    }
  ]
}
```

Mismatches indicate:
- Potential bugs in one or both tools
- Bounded vs unbounded analysis differences
- Timeout or resource limitations
- Edge cases in SVA semantics

## Integration Examples

### Python Integration

```python
import subprocess
import json

def verify_with_vcformal(sva1: str, sva2: str) -> bool:
    """Check if sva1 and sva2 are equivalent using VCFormal."""
    result = subprocess.run(
        ["sva-vcformal-implication", "equivalent",
         "--sva1", sva1, "--sva2", sva2,
         "--json-output"],
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        raise RuntimeError(f"VCFormal verification failed: {result.stderr}")

    data = json.loads(result.stdout)
    return data["equivalent"]

# Example usage
try:
    equiv = verify_with_vcformal(
        "req |-> ##1 ack",
        "req |-> ##[1:1] ack"
    )
    print(f"Equivalence: {equiv}")
except Exception as e:
    print(f"Error: {e}")
```

### Automated Testing Pipeline

```bash
#!/bin/bash
# test_properties.sh - Batch verification script

INPUT_DIR="test_cases"
OUTPUT_DIR="results"

mkdir -p "$OUTPUT_DIR"

for test_file in "$INPUT_DIR"/*.json; do
    base_name=$(basename "$test_file" .json)
    echo "Processing: $base_name"

    sva-vcformal-implication cross-validate \
        --input-file "$test_file" \
        --output-file "$OUTPUT_DIR/${base_name}_results.json" \
        --ebmc-path /usr/local/bin/ebmc \
        --ebmc-depth 30 \
        --require-ebmc

    if [ $? -eq 0 ]; then
        echo "✓ $base_name completed successfully"
    else
        echo "✗ $base_name failed"
    fi
done
```

## Performance Considerations

1. **Timeout tuning**: Complex properties may require timeouts exceeding 300 seconds
2. **Engine selection**: VCFormal automatically selects engines; observe which works best for your properties
3. **Parallelization**: Cross-validation tasks can be parallelized by splitting input files
4. **Resource monitoring**: VCFormal can be memory-intensive; monitor system resources

## Limitations

1. **Commercial license requirement**: VC Formal requires valid Cadence licenses
2. **Platform dependency**: VCFormal availability varies by operating system and version
3. **Inconclusive results**: Some properties may timeout or yield INCONCLUSIVE results
4. **SVA subset support**: Not all SVA constructs are supported by formal engines

## Troubleshooting

### VCFormal Not Found

```bash
# Check VCFormal installation
which vcf

# Specify explicit path
sva-vcformal-implication --vcf-path /opt/cadence/bin/vcf check ...
```

### License Issues

Ensure Cadence license server is accessible:

```bash
export LM_LICENSE_FILE=port@license_server
sva-vcformal-implication check ...
```

### Cross-Validation Discrepancies

If VCFormal and EBMC disagree:

1. Increase EBMC depth: `--ebmc-depth 50`
2. Check for bounded vs unbounded differences
3. Manually inspect the property pair
4. Report as potential tool bug if verified manually

### Timeout Optimization

For slow verifications:

```bash
sva-vcformal-implication --timeout 1200 check ...
```

Or simplify properties by decomposition.

## References

- Cadence VC Formal User Guide (proprietary documentation)
- Formal Verification: An Essential Toolkit for Modern VLSI Design (2nd Edition)
- IEEE 1800-2017 SystemVerilog Standard
- Cross-validation methodology: "Comparing Formal Verification Tools" (DAC 2018)
