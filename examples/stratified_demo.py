#!/usr/bin/env python3
"""
Stratified Sampling Demonstration

Shows the difference between random and stratified sampling modes.
"""

from sva_toolkit.gen import SVASynthesizer, StratifiedGenerator
import random
import json
from collections import Counter


def analyze_coverage(properties, title="Dataset"):
    """Analyze and display coverage statistics."""

    constructs = {
        '|->': 0, '|=>': 0, ' and ': 0, ' or ': 0, 'until': 0,
        '##': 0, '[*': 0, '[=': 0, '[->': 0,
        '$rose': 0, '$fell': 0, '$stable': 0, '$changed': 0, '$past': 0,
        '$onehot': 0, '$onehot0': 0, '$isunknown': 0, '$countones': 0,
        '&&': 0, '||': 0, '!': 0,
        '==': 0, '!=': 0, '===': 0, '!==': 0, '>': 0, '<': 0, '>=': 0, '<=': 0,
        ' + ': 0, ' - ': 0, ' * ': 0, ' / ': 0, ' % ': 0,
        ' & ': 0, ' | ': 0, ' ^ ': 0, '^~': 0, '~': 0,
    }

    for prop in properties:
        sva = prop.sva_code if hasattr(prop, 'sva_code') else prop['sva']
        for construct in constructs.keys():
            if construct in sva:
                constructs[construct] += 1

    covered = sum(1 for count in constructs.values() if count > 0)
    total = len(constructs)
    coverage_pct = (covered / total) * 100

    print(f"\n{title}:")
    print("-" * 80)
    print(f"  Total properties: {len(properties)}")
    print(f"  Coverage: {covered}/{total} constructs ({coverage_pct:.1f}%)")

    missing = [c for c, count in constructs.items() if count == 0]
    if missing:
        print(f"  Missing: {len(missing)} constructs")
        print(f"    {', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else ""))
    else:
        print(f"  ✓ COMPLETE COVERAGE!")

    return covered, total, coverage_pct


def demo_random_vs_stratified():
    """Compare random vs stratified sampling."""

    print("=" * 80)
    print("  RANDOM VS STRATIFIED SAMPLING COMPARISON")
    print("=" * 80)

    signals = ['req', 'ack', 'valid', 'ready', 'data']

    # Random mode - Generate 500 samples
    print("\n[1] RANDOM MODE - 500 samples")
    print("-" * 80)

    random.seed(42)
    random_synth = SVASynthesizer(signals=signals, max_depth=2)

    random_props = []
    for i in range(500):
        prop = random_synth.synthesize(f'random_{i}')
        random_props.append(prop)

    random_covered, random_total, random_pct = analyze_coverage(
        random_props,
        "Random Sampling (500 samples)"
    )

    # Stratified mode - 10 samples per construct
    print("\n[2] STRATIFIED MODE - 10 samples per construct")
    print("-" * 80)

    random.seed(42)
    stratified_gen = StratifiedGenerator(
        signals=signals,
        max_depth=2,
        samples_per_construct=10
    )

    stratified_props = stratified_gen.generate_stratified_dataset()

    stratified_covered, stratified_total, stratified_pct = analyze_coverage(
        stratified_props,
        f"Stratified Sampling ({len(stratified_props)} samples)"
    )

    # Comparison
    print("\n" + "=" * 80)
    print("  COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<30} | {'Random (500)':<20} | {'Stratified (~470)':<20}")
    print("-" * 80)
    print(f"{'Constructs Covered':<30} | {random_covered}/{random_total} ({random_pct:.1f}%){'':<8} | {stratified_covered}/{stratified_total} ({stratified_pct:.1f}%){'':<8}")
    print(f"{'Efficiency (samples/coverage)':<30} | {500/random_covered if random_covered > 0 else 0:.1f} samples/construct{'':<3} | {len(stratified_props)/stratified_covered if stratified_covered > 0 else 0:.1f} samples/construct")

    print("\n" + "=" * 80)
    print("  KEY INSIGHTS")
    print("=" * 80)
    print("""
1. Random Sampling:
   - Probabilistic coverage
   - May miss rare constructs
   - Good for diverse, realistic patterns
   - Suitable for large datasets

2. Stratified Sampling:
   - GUARANTEED 100% coverage
   - Equal representation of all constructs
   - More efficient for complete coverage
   - Ideal for training data

Recommendation:
   - Use STRATIFIED for initial training data (guaranteed coverage)
   - Use RANDOM for additional diversity and realism
   - Combine both approaches for best results
""")


def demo_stratified_usage():
    """Show different ways to use stratified sampling."""

    print("\n" + "=" * 80)
    print("  STRATIFIED SAMPLING USAGE EXAMPLES")
    print("=" * 80)

    print("""
CLI Usage:
----------

# Generate with default settings (50 samples per construct)
$ sva-gen generate --mode stratified -j -o dataset.json

# Generate with custom samples per construct
$ sva-gen generate --mode stratified --samples-per-construct 100 -j -o large_dataset.json

# With specific signals
$ sva-gen generate --mode stratified --preset axi -j -o axi_dataset.json

# With random seed for reproducibility
$ sva-gen generate --mode stratified --seed 42 -j -o reproducible.json


Programmatic Usage:
-------------------

from sva_toolkit.gen import StratifiedGenerator

# Create generator
generator = StratifiedGenerator(
    signals=['req', 'ack', 'valid', 'ready'],
    max_depth=2,
    samples_per_construct=50
)

# Generate dataset with guaranteed coverage
properties = generator.generate_stratified_dataset()

# Export to JSON
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

with open('stratified_dataset.json', 'w') as f:
    json.dump(data, f, indent=2)


Hybrid Approach (Recommended for Training):
-------------------------------------------

# 1. Generate stratified base (guaranteed coverage)
$ sva-gen generate --mode stratified --samples-per-construct 100 -j -o train_base.json

# 2. Generate random samples for diversity
$ sva-gen generate --mode random -n 10000 -j -o train_diverse.json --seed 42

# 3. Combine datasets
# Result: Complete coverage + diverse patterns
""")


def demo_coverage_scaling():
    """Show how coverage scales with samples per construct."""

    print("\n" + "=" * 80)
    print("  COVERAGE SCALING ANALYSIS")
    print("=" * 80)

    signals = ['req', 'ack', 'valid', 'ready']

    samples_per_construct_values = [5, 10, 25, 50, 100]

    print("\nGenerating datasets with varying samples per construct...\n")
    print(f"{'Samples/Construct':<20} | {'Total Samples':<15} | {'Coverage':<15}")
    print("-" * 80)

    for spc in samples_per_construct_values:
        random.seed(42)
        gen = StratifiedGenerator(
            signals=signals,
            max_depth=2,
            samples_per_construct=spc
        )

        # Generate without printing progress
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        props = gen.generate_stratified_dataset()
        sys.stdout = old_stdout

        _, _, coverage = analyze_coverage(props, "")

        bar_length = int(coverage / 2)
        bar = "█" * bar_length
        print(f"{spc:<20} | {len(props):<15} | {coverage:>5.1f}%  {bar}")

    print("\n✓ Stratified mode guarantees 100% coverage regardless of sample count")


if __name__ == "__main__":
    demo_random_vs_stratified()
    demo_stratified_usage()
    demo_coverage_scaling()

    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)
    print("""
Stratified sampling provides GUARANTEED complete coverage of all SVA constructs,
making it ideal for creating high-quality training datasets for ML models.

Benefits:
  ✓ 100% coverage guaranteed
  ✓ Predictable dataset composition
  ✓ Efficient (no wasted samples on over-represented constructs)
  ✓ Balanced representation of rare and common constructs

Use stratified mode for your training data generation!
""")
    print("=" * 80 + "\n")
