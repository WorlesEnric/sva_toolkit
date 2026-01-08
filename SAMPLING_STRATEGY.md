# SVA Generation Sampling Strategy

## Overview

The SVA generator uses **probabilistic recursive sampling** to compose properties from the SVA syntax set. This document explains how sampling works and analyzes coverage guarantees.

## Hierarchical Sampling Process

### Level 1: Property Generation (Top Level)

```python
def generate_property(depth=0):
    choice = random.random()  # Uniform [0, 1]

    if choice < 0.75:           # 75% - Implication
        op = random.choice(["|->", "|=>"])  # Uniform choice
        return Implication(generate_sequence(), op, generate_sequence())

    elif choice < 0.82:         # 7% - Property AND/OR
        op = random.choice(["and", "or"])
        return PropertyBinary(generate_sequence(), op, generate_sequence())

    elif choice < 0.88:         # 6% - Until operators
        op = random.choice(["until", "until_with"])
        return PropertyUntil(generate_sequence(), op, generate_sequence())

    elif choice < 0.92:         # 4% - If-else
        return PropertyIfElse(generate_bool(), generate_sequence(), ...)

    else:                        # 8% - Direct sequence or NOT
        prop = generate_sequence()
        if random.random() < 0.2:  # 20% of 8% = 1.6% overall
            return NotProperty(prop)
        return prop

    # 10% chance at depth=0 to wrap in disable iff
```

**Probability Distribution:**
- Implication (`|->`, `|=>`): **75%**
  - Overlapping `|->`: **37.5%**
  - Non-overlapping `|=>`: **37.5%**
- Property binary (`and`, `or`): **7%**
  - `and`: **3.5%**
  - `or`: **3.5%**
- Until (`until`, `until_with`): **6%**
  - `until`: **3%**
  - `until_with`: **3%**
- If-else: **4%**
- NOT property: **1.6%**
- DisableIff (wrapper): **~10%** at top level

### Level 2: Sequence Generation

```python
def generate_sequence(depth):
    choice = random.random()

    if depth >= max_depth or choice < 0.45:  # 45% base case
        return generate_bool()

    elif choice < 0.65:         # 20% - Sequence delay
        delay = random_delay()  # ##1, ##[1:3], ##[0:$]
        return SequenceDelay(generate_sequence(), delay, generate_sequence())

    elif choice < 0.80:         # 15% - Sequence repeat
        op = random.choice(["[*", "[=", "[->"])
        count = random_repeat_count()
        return SequenceRepeat(generate_sequence(), op, count)

    elif choice < 0.90:         # 10% - Binary sequence ops
        op = random.choice(["intersect", "throughout", "and", "or"])
        return SequenceBinary(generate_sequence(), op, generate_sequence())

    else:                        # 10% - first_match or bool
        if enable_advanced and random.random() < 0.3:  # 3% overall
            return SequenceFirstMatch(generate_sequence())
        return generate_bool()
```

**Probability Distribution (at depth < max_depth):**
- Base case (boolean): **45%**
- Delay (`##`): **20%**
- Repeat (`[*`, `[=`, `[->`): **15%**
  - Each operator: **~5%**
- Binary ops (`intersect`, `throughout`, `and`, `or`): **10%**
  - Each operator: **~2.5%**
- `first_match`: **3%**

### Level 3: Boolean Generation

```python
def generate_bool(depth):
    choice = random.random()

    if depth >= max_depth or choice < 0.25:  # 25% leaf
        leaf_choice = random.random()
        if leaf_choice < 0.4:                 # 10% overall
            func = random.choice([
                "$rose", "$fell", "$stable", "$changed",
                "$onehot", "$onehot0", "$isunknown", "$countones"
            ])  # 8 functions, ~1.25% each
            return UnarySysFunction(func, get_random_signal())
        elif leaf_choice < 0.5:               # 2.5% overall
            return SequenceEnded(generate_sequence())
        else:                                  # 12.5% overall
            return get_random_signal()

    elif choice < 0.4:          # 15% - Logical NOT
        return UnaryOp("!", generate_bool())

    elif choice < 0.7:          # 30% - Comparison
        op = random.choice([">", "<", ">=", "<=", "==", "!=", "===", "!=="])
        # 8 operators, ~3.75% each
        return BinaryOp(generate_expr(), op, generate_expr())

    else:                        # 30% - Logical operation
        op = random.choice(["&&", "||"])  # 15% each
        return BinaryOp(generate_bool(), op, generate_bool())
```

**Probability Distribution:**
- System functions (total): **10%**
  - `$rose`: **~1.25%**
  - `$fell`: **~1.25%**
  - `$stable`: **~1.25%**
  - `$changed`: **~1.25%**
  - `$onehot`: **~1.25%**
  - `$onehot0`: **~1.25%**
  - `$isunknown`: **~1.25%**
  - `$countones`: **~1.25%**
- `sequence.ended`: **2.5%**
- Bare signal: **12.5%**
- Logical NOT (`!`): **15%**
- Comparison operators (total): **30%**
  - Each of 8 operators: **~3.75%**
- Logical operators: **30%**
  - `&&`: **15%**
  - `||`: **15%**

### Level 4: Expression Generation

```python
def generate_expr(depth):
    if depth >= max_depth or random.random() > 0.7:  # 30% leaf
        if random.random() < 0.2:  # 6% overall
            return PastFunction(get_random_signal(), random.randint(1, 3))
        return get_random_signal()  # 24% overall

    choice = random.random()

    if choice < 0.15:           # 10.5% - Unary operation
        op = random.choice(["~", "-", "+"])  # ~3.5% each
        return UnaryOp(op, generate_expr())

    elif choice < 0.25:         # 7% - Ternary (if enabled)
        return TernaryOp(generate_bool(), generate_expr(), generate_expr())

    else:                        # 52.5% - Binary operation
        op = random.choice(["+", "-", "*", "/", "%", "&", "|", "^", "^~", "~^"])
        # 10 operators, ~5.25% each
        return BinaryOp(generate_expr(), op, generate_expr())
```

## Coverage Analysis

### Expected Probabilities of Each Construct

| Construct | Base Probability | Notes |
|-----------|-----------------|-------|
| **Property Level** | | |
| `\|->` | 37.5% | Very common |
| `\|=>` | 37.5% | Very common |
| `and` (property) | 3.5% | Moderate |
| `or` (property) | 3.5% | Moderate |
| `until` | 3% | Moderate |
| `until_with` | 3% | Moderate |
| `if...else` | 4% | Moderate |
| `not` | 1.6% | Rare |
| `disable iff` | ~10% | Wrapper only |
| **Sequence Level** | | |
| `##` (delay) | 20% × depth_factor | Common |
| `[*` | ~5% × depth_factor | Moderate |
| `[=` | ~5% × depth_factor | Moderate |
| `[->` | ~5% × depth_factor | Moderate |
| `intersect` | ~2.5% × depth_factor | Rare |
| `throughout` | ~2.5% × depth_factor | Rare |
| `first_match` | 3% × depth_factor | Rare |
| **Boolean Level** | | |
| `$rose` | ~1.25% × depth_factor | Rare |
| `$fell` | ~1.25% × depth_factor | Rare |
| `$stable` | ~1.25% × depth_factor | Rare |
| `$changed` | ~1.25% × depth_factor | Rare |
| `$onehot` | ~1.25% × depth_factor | Rare |
| `$onehot0` | ~1.25% × depth_factor | Rare |
| `$isunknown` | ~1.25% × depth_factor | Rare |
| `$countones` | ~1.25% × depth_factor | Rare |
| `.ended` | ~2.5% × depth_factor | Rare |
| `!` (logical) | 15% | Common |
| `&&` | 15% | Common |
| `\|\|` | 15% | Common |
| `>`, `<`, `>=`, `<=` | ~3.75% each | Moderate |
| `==`, `!=`, `===`, `!==` | ~3.75% each | Moderate |
| **Expression Level** | | |
| `$past` | ~6% × depth_factor | Moderate |
| `~`, `-`, `+` (unary) | ~3.5% each × depth_factor | Moderate |
| `? :` (ternary) | ~7% × depth_factor | Moderate |
| Arithmetic/bitwise ops | ~5% each × depth_factor | Moderate |

### Coverage Guarantee

**Question: Does this guarantee every construct appears in a sufficiently large dataset?**

**Answer: NO - Not guaranteed with current purely random sampling.**

#### Why Not Guaranteed?

1. **Probabilistic Nature**: Even with low probabilities, rare events can be missed in finite samples
2. **Depth Dependency**: Constructs at deeper levels have compound probabilities that decrease exponentially
3. **No Tracking**: No mechanism to ensure minimum coverage

#### Expected Coverage

Using **Poisson approximation**, the probability of seeing a construct with probability `p` at least once in `n` samples:

```
P(X ≥ 1) = 1 - e^(-n*p)
```

For a construct with probability `p = 0.01` (1%):
- n = 100: P(X ≥ 1) = 63%
- n = 500: P(X ≥ 1) = 99.3%
- n = 1000: P(X ≥ 1) = 99.995%

For a construct with probability `p = 0.001` (0.1%):
- n = 1000: P(X ≥ 1) = 63%
- n = 5000: P(X ≥ 1) = 99.3%
- n = 10000: P(X ≥ 1) = 99.995%

### Minimum Dataset Sizes for Coverage

To achieve **99.9% probability** of seeing each construct at least once:

| Rarest Construct Probability | Minimum Dataset Size |
|------------------------------|---------------------|
| 1% | ~700 samples |
| 0.5% | ~1,400 samples |
| 0.1% | ~7,000 samples |
| 0.05% | ~14,000 samples |

**Current rarest constructs:**
- Individual system functions: **~1.25% at depth 0**, decreasing with depth
- `sequence.ended`: **~2.5% at depth 0**
- Individual sequence binary ops: **~2.5% at depth 0**

**Recommendation for full coverage: Generate at least 5,000-10,000 samples**

## Improving Coverage Guarantees

### Option 1: Stratified Sampling

Generate a fixed number of examples for each construct category:

```python
def generate_stratified_dataset(samples_per_construct=100):
    dataset = []

    # Ensure each implication type
    for _ in range(samples_per_construct):
        dataset.append(generate_with_constraint(must_have='|->'))
        dataset.append(generate_with_constraint(must_have='|=>'))

    # Ensure each system function
    for func in ["$rose", "$fell", "$stable", "$changed", ...]:
        for _ in range(samples_per_construct):
            dataset.append(generate_with_constraint(must_have=func))

    return dataset
```

### Option 2: Coverage-Aware Generation

Track what's been generated and boost probabilities of rare constructs:

```python
class CoverageAwareGenerator:
    def __init__(self):
        self.coverage = {construct: 0 for construct in ALL_CONSTRUCTS}
        self.min_coverage = 100

    def adjust_probability(self, construct):
        # Boost probability if under-represented
        if self.coverage[construct] < self.min_coverage:
            return original_prob * 2.0
        return original_prob
```

### Option 3: Explicit Coverage Mode

Add a mode that systematically covers all constructs:

```bash
# Generate with guaranteed coverage
sva-gen generate -n 1000 --coverage-mode full -j -o dataset.json
```

## Current Recommendation

For training data generation with **current random sampling**:

1. **Minimum dataset size: 10,000 samples** for >99% coverage of all constructs
2. **Recommended size: 20,000-50,000 samples** for robust coverage including combinations
3. **Use different random seeds** for train/val/test splits
4. **Verify coverage** by analyzing the generated dataset:

```python
# Coverage analysis
import json
from collections import Counter

with open('dataset.json') as f:
    data = json.load(f)

# Count construct occurrences
constructs = Counter()
for prop in data['properties']:
    sva = prop['sva']
    # Count keywords
    if '|=>' in sva: constructs['|=>'] += 1
    if '|->' in sva: constructs['|->'] += 1
    if '$rose' in sva: constructs['$rose'] += 1
    # ... etc

print("Coverage report:")
for construct, count in sorted(constructs.items()):
    print(f"  {construct}: {count} occurrences")
```

## Conclusion

- **Current sampling**: Probabilistic, hierarchical, no guarantees
- **Coverage expectation**: 99%+ with 10,000+ samples
- **Recommendation**: Generate large datasets (20K-50K) for robust coverage
- **Future improvement**: Implement stratified or coverage-aware sampling for guaranteed coverage
