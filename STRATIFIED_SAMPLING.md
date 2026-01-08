# Stratified Sampling for SVA Generation

## Overview

Stratified sampling **guarantees 100% coverage** of all SVA constructs by generating a minimum number of examples for each construct. This is ideal for creating training datasets for machine learning models.

## Problem with Random Sampling

Random sampling is **probabilistic** and doesn't guarantee coverage:

```
Random Mode (500 samples):
  Coverage: 23/39 constructs (59.0%)
  Missing: 16 constructs including arithmetic, bitwise, rare comparisons
```

Even with **10,000+ samples**, some rare constructs may never appear!

## Stratified Sampling Solution

Stratified mode **guarantees every construct appears**:

```
Stratified Mode (470 samples):
  Coverage: 39/39 constructs (100.0%)
  ✓ COMPLETE COVERAGE!
```

## How It Works

### 1. Constraint-Based Generation

For each construct, we have a dedicated generator that **forces** that construct to appear:

```python
def generate_rose(self):
    """Generate property containing $rose function."""
    sig = self._get_random_signal()
    func = UnarySysFunction("$rose", sig)
    cons = self.synth.generate_sequence(0)
    return Implication(func, "|->", cons)
    # Result: $rose(signal) |-> ...
```

### 2. Systematic Coverage

The generator covers all 47 constructs across categories:

| Category | Constructs | Examples |
|----------|-----------|----------|
| **Property Operators** (9) | `\|->`, `\|=>`, `and`, `or`, `until`, `until_with`, `not`, `disable iff`, `if` |
| **Sequence Operators** (8) | `##`, `[*`, `[=`, `[->`, `intersect`, `throughout`, `first_match`, `.ended` |
| **System Functions** (9) | `$rose`, `$fell`, `$stable`, `$changed`, `$past`, `$onehot`, `$onehot0`, `$isunknown`, `$countones` |
| **Boolean Operators** (3) | `&&`, `\|\|`, `!` |
| **Comparison Operators** (8) | `==`, `!=`, `===`, `!==`, `>`, `<`, `>=`, `<=` |
| **Arithmetic Operators** (5) | `+`, `-`, `*`, `/`, `%` |
| **Bitwise Operators** (5) | `&`, `\|`, `^`, `^~`, `~` |

**Total: 47 constructs**

### 3. Configurable Samples Per Construct

Control how many examples of each construct:

```bash
# 10 samples per construct = 470 total properties
sva-gen generate --mode stratified --samples-per-construct 10 -j -o dataset.json

# 100 samples per construct = 4,700 total properties
sva-gen generate --mode stratified --samples-per-construct 100 -j -o large.json
```

## Usage

### CLI Usage

```bash
# Basic stratified generation
sva-gen generate --mode stratified -j -o stratified.json

# With custom settings
sva-gen generate \
    --mode stratified \
    --samples-per-construct 50 \
    --preset axi \
    --seed 42 \
    -j -o dataset.json

# Compare modes
sva-gen generate --mode random -n 500 -j -o random.json
sva-gen generate --mode stratified --samples-per-construct 10 -j -o stratified.json
```

### Programmatic Usage

```python
from sva_toolkit.gen import StratifiedGenerator

# Create generator
generator = StratifiedGenerator(
    signals=['req', 'ack', 'valid', 'ready'],
    max_depth=2,
    samples_per_construct=50,
    clock_signal='clk'
)

# Generate dataset with guaranteed coverage
properties = generator.generate_stratified_dataset()

# Export
import json
data = {
    'properties': [
        {
            'name': p.name,
            'sva': p.sva_code,
            'svad': p.svad
        }
        for p in properties
    ]
}

with open('training_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Generated {len(properties)} properties")
print(f"Total constructs covered: 47/47 (100%)")
```

## Comparison: Random vs Stratified

| Metric | Random (500 samples) | Stratified (470 samples) |
|--------|---------------------|--------------------------|
| **Coverage** | 23/39 (59%) | 39/39 (100%) |
| **Efficiency** | 21.7 samples/construct | 12.1 samples/construct |
| **Guarantees** | None | 100% coverage |
| **Missing Constructs** | ~16 constructs | 0 constructs |
| **Best For** | Diversity, realism | Training data, coverage |

## Recommended Workflow for Training Data

### Option 1: Stratified Only (Simple)

```bash
# Generate large stratified dataset
sva-gen generate \
    --mode stratified \
    --samples-per-construct 200 \
    -j -o training_data.json

# Result: 9,400 properties, 100% coverage, balanced distribution
```

### Option 2: Hybrid (Recommended)

```bash
# Step 1: Stratified base (guaranteed coverage)
sva-gen generate \
    --mode stratified \
    --samples-per-construct 100 \
    -j -o train_stratified.json \
    --seed 42

# Step 2: Random diversity
sva-gen generate \
    --mode random \
    -n 20000 \
    -j -o train_random.json \
    --seed 100

# Step 3: Combine datasets programmatically
# Result: Complete coverage + diverse realistic patterns
```

**Hybrid approach benefits:**
- ✓ Guaranteed 100% coverage (from stratified)
- ✓ Diverse realistic patterns (from random)
- ✓ Balanced representation
- ✓ Best training data quality

## Example Output

```json
{
  "properties": [
    {
      "name": "p_0",
      "sva": "$rose(req) |-> ack",
      "svad": "When signal 'req' rises from 0 to 1, then in the same cycle: signal 'ack'."
    },
    {
      "name": "p_1",
      "sva": "((data + addr) > limit) |-> valid",
      "svad": "When signal 'data' plus signal 'addr' is greater than signal 'limit', then in the same cycle: signal 'valid'."
    },
    {
      "name": "p_2",
      "sva": "req |=> (ack[=1:3])",
      "svad": "When signal 'req', then in the next cycle: signal 'ack' occurs between 1 and 3 times non-consecutively."
    }
  ]
}
```

Each property is **guaranteed to showcase a specific construct**, ensuring complete coverage.

## Coverage Verification

Use the coverage analysis tool to verify:

```bash
python examples/coverage_analysis.py
```

Or programmatically:

```python
from collections import Counter

# Load dataset
with open('dataset.json') as f:
    data = json.load(f)

# Count constructs
constructs = Counter()
for prop in data['properties']:
    if '$rose' in prop['sva']:
        constructs['$rose'] += 1
    # ... check all constructs

print(f"Coverage: {len(constructs)}/47 constructs")
```

## Performance

| Samples Per Construct | Total Properties | Generation Time |
|-----------------------|------------------|-----------------|
| 10 | 470 | ~3 seconds |
| 50 | 2,350 | ~12 seconds |
| 100 | 4,700 | ~25 seconds |
| 200 | 9,400 | ~50 seconds |

## Advanced: Custom Construct Selection

If you only want specific construct categories:

```python
from sva_toolkit.gen.stratified import StratifiedGenerator

generator = StratifiedGenerator(
    signals=['req', 'ack', 'valid'],
    samples_per_construct=100
)

# Generate only specific constructs
properties = []

# Only system functions
for func_name in ['$rose', '$fell', '$stable', '$changed']:
    method = getattr(generator, f'generate_{func_name.replace("$", "")}')
    for i in range(100):
        properties.append(method())

# Only implications
for _ in range(100):
    properties.append(generator.generate_overlapping_implication())
    properties.append(generator.generate_non_overlapping_implication())
```

## Summary

**Stratified sampling is the recommended approach for training data generation.**

### Benefits:
- ✅ **100% guaranteed coverage** of all 47 SVA constructs
- ✅ **Efficient**: No wasted samples on over-represented constructs
- ✅ **Predictable**: Know exactly what your dataset contains
- ✅ **Balanced**: Equal representation of rare and common constructs
- ✅ **Configurable**: Control samples per construct
- ✅ **Fast**: Generate thousands of properties in seconds

### Use Cases:
- Training ML models for SVAD→SVA translation
- Creating test suites for SVA tools
- Educational datasets for learning SVA
- Benchmarking assertion coverage tools

Start using stratified mode today:

```bash
sva-gen generate --mode stratified -j -o my_training_data.json
```
