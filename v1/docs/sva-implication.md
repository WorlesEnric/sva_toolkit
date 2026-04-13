# SVA Implication Checker (`sva-implication`)

## Overview

The SVA Implication Checker is a formal verification tool that employs bounded model checking (BMC) to determine logical implication relationships between pairs of SystemVerilog Assertions. The tool utilizes EBMC (Efficient Bounded Model Checker) as its backend verification engine to establish whether one property logically implies another, or whether two properties are semantically equivalent.

## Theoretical Foundation

In formal verification, an implication relationship `P → Q` (read "P implies Q") holds if every execution trace that satisfies property P also satisfies property Q. Equivalence `P ⇔ Q` requires bidirectional implication: both `P → Q` and `Q → P` must hold. These relationships are fundamental for:

1. **Property refinement**: Verifying that a concrete implementation satisfies an abstract specification
2. **Assertion optimization**: Eliminating redundant properties from verification suites
3. **Correctness checking**: Validating that generated assertions match reference specifications
4. **Property hierarchy construction**: Building lattices of assertion strength for compositional verification

The checker transforms SVA properties into equivalent transition system representations suitable for BMC analysis. By systematically exploring all execution paths up to a bounded depth, the tool can either prove implication (if no counterexample exists within the bound) or provide a concrete counterexample trace demonstrating violation.

## Architecture

The verification workflow consists of the following stages:

1. **Property preprocessing**: SVA code is parsed and normalized into canonical form
2. **Module synthesis**: A wrapper SystemVerilog module is generated containing both properties as assertions
3. **Negation transformation**: For checking `P → Q`, the tool verifies that `¬(P ∧ ¬Q)` is valid
4. **BMC invocation**: EBMC analyzes the generated module up to the specified depth bound
5. **Result interpretation**: Tool output is parsed to determine IMPLIES, NOT_IMPLIES, or ERROR status
6. **Counterexample extraction**: If implication fails, a witness trace is extracted from EBMC output

## Soundness and Completeness

**Soundness**: If the checker reports IMPLIES, the implication holds for all traces up to the specified bound.

**Completeness**: The checker is incomplete due to bounded exploration. A NOT_IMPLIES result is definitive (counterexample provided), but absence of a counterexample within the bound does not guarantee the implication holds for all depths. In practice, bounds of 20-30 cycles are sufficient for most hardware properties.

## Use Cases

1. **LLM-generated assertion validation**: Verifying that machine learning models produce semantically correct SVA properties
2. **Regression testing**: Ensuring refactored properties maintain semantic equivalence
3. **Property coverage analysis**: Identifying subsumed assertions in large verification suites
4. **Specification conformance**: Checking implementation-level assertions against high-level requirements
5. **Dataset quality assurance**: Validating SVA training datasets for machine learning applications

## Command Reference

### Check Implication

Verify whether antecedent implies consequent:

```bash
sva-implication check \
  --antecedent "req |-> ##[1:2] ack" \
  --consequent "req |-> ##1 ack"
```

Expected output: SUCCESS (consequent is weaker, so implication holds).

### Check Implication with Verbose Output

Include detailed verification logs and counterexamples:

```bash
sva-implication check \
  --antecedent "req |-> ##1 ack" \
  --consequent "req |-> ##2 ack" \
  --verbose
```

If implication fails, the counterexample trace will be displayed.

### Check Equivalence

Determine if two properties are semantically equivalent:

```bash
sva-implication equivalent \
  --sva1 "a ##1 b" \
  --sva2 "a ##[1:1] b"
```

Expected output: EQUIVALENT (both properties specify the same behavior).

### Determine Full Relationship

Analyze bidirectional implication to classify the relationship:

```bash
sva-implication relationship \
  --sva1 "req |-> ##[1:3] ack" \
  --sva2 "req |-> ##2 ack"
```

Output possibilities:
- EQUIVALENT: Bidirectional implication
- SVA1 STRONGER: Only SVA1 → SVA2 holds
- SVA2 STRONGER: Only SVA2 → SVA1 holds
- NO RELATIONSHIP: Neither direction holds

### Batch Equivalence Checking

Process multiple SVA pairs from a JSON file:

```bash
sva-implication batch-equivalent \
  --input-file pairs.json \
  --output-file results.json \
  --verbose
```

Input JSON format:
```json
[
  {"id": "pair1", "sva1": "...", "sva2": "..."},
  {"id": "pair2", "sva1": "...", "sva2": "..."}
]
```

Output JSON format:
```json
[
  {
    "id": "pair1",
    "result": "EQUIVALENT",
    "equivalent": true,
    "message": "Properties are equivalent"
  }
]
```

### JSON Output Format

Export results in machine-readable JSON:

```bash
sva-implication check \
  --antecedent "req" \
  --consequent "req && valid" \
  --json-output
```

### Custom Configuration

Adjust verification parameters:

```bash
sva-implication \
  --ebmc-path /custom/path/to/ebmc \
  --depth 30 \
  --work-dir /tmp/verification \
  --keep-files \
  check --antecedent "..." --consequent "..."
```

Parameters:
- `--ebmc-path`: Path to EBMC binary
- `--depth`: BMC unrolling depth (default: 20)
- `--work-dir`: Directory for temporary verification files
- `--keep-files`: Preserve generated SystemVerilog files for debugging

## Integration Examples

### Python Integration for Dataset Validation

```python
import subprocess
import json

def check_equivalence(sva1, sva2):
    result = subprocess.run(
        ["sva-implication", "equivalent",
         "--sva1", sva1, "--sva2", sva2,
         "--json-output"],
        capture_output=True,
        text=True
    )
    data = json.loads(result.stdout)
    return data["equivalent"]

# Validate generated assertion against reference
generated = "req |-> ##[1:2] ack"
reference = "req |-> ##1 ack"
if check_equivalence(generated, reference):
    print("Assertion validated successfully")
else:
    print("Semantic mismatch detected")
```

### Batch Processing with Error Handling

```bash
#!/bin/bash
while IFS='|' read -r id sva1 sva2; do
  echo "Checking pair: $id"
  sva-implication equivalent \
    --sva1 "$sva1" \
    --sva2 "$sva2" \
    --json-output 2>&1 | tee -a results.log
done < pairs.txt
```

## Performance Considerations

1. **Depth selection**: Higher depths increase verification time exponentially; start with depth 15-20
2. **Property complexity**: Properties with long delay ranges require deeper bounds
3. **Signal count**: Larger signal sets increase state space; consider abstraction when possible
4. **Parallelization**: Batch operations can be parallelized across multiple cores

## Limitations

1. **Bounded verification**: Cannot prove universal validity, only falsify or verify within bounds
2. **EBMC dependency**: Requires functional EBMC installation with appropriate licenses
3. **Temporal depth**: Very deep temporal properties may require impractically large bounds
4. **Expressiveness**: Limited to SVA subset supported by EBMC's translation pipeline

## Troubleshooting

### EBMC Not Found

```bash
export EBMC_PATH=/path/to/ebmc
sva-implication --ebmc-path $EBMC_PATH check ...
```

### Timeout Issues

Increase depth or simplify properties:

```bash
sva-implication --depth 50 check ...
```

### Debugging Generated Code

Preserve temporary files for manual inspection:

```bash
sva-implication --keep-files --work-dir ./debug check ...
ls ./debug/*.sv
```

## References

- EBMC: https://www.cprover.org/ebmc/
- Bounded Model Checking: Clarke, E., Biere, A., Raimi, R., & Zhu, Y. (2001)
- IEEE 1800-2017 SystemVerilog Standard, Section 16: Assertions
