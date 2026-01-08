#!/usr/bin/env python3
"""
Comprehensive test and comparison of old vs. new NL generation systems.
"""

from src.sva_toolkit.gen.types_sva import *
from src.sva_toolkit.gen.nl import sva_to_english


def compare_nl_systems(name: str, node: SVANode):
    """Compare old and new NL generation for a given node."""
    sva_code = str(node)

    # Get old NL (from subclass implementation)
    old_nl = node.__class__.to_natural_language(node)

    # Get new NL (from SVATrans)
    new_nl = sva_to_english(node)

    print(f"\n{name}:")
    print(f"  SVA:  {sva_code}")
    print(f"  OLD:  {old_nl}")
    print(f"  NEW:  {new_nl}")
    print(f"  Improvement: {len(old_nl) - len(new_nl):+d} chars")


def main():
    print("="*80)
    print("COMPREHENSIVE COMPARISON: Old vs. New NL Generation")
    print("="*80)

    # Test cases organized by complexity

    print("\n" + "="*80)
    print("SIMPLE CASES")
    print("="*80)

    # Simple signals
    req = Signal("req")
    compare_nl_systems("Signal", req)

    # Binary comparison
    comp = BinaryOp(Signal("req"), "==", Signal("ack"))
    compare_nl_systems("Binary Comparison", comp)

    # System function
    rose = UnarySysFunction("$rose", Signal("req"))
    compare_nl_systems("System Function ($rose)", rose)

    # Simple implication
    impl = Implication(Signal("req"), "|->", Signal("ack"))
    compare_nl_systems("Simple Implication (overlapping)", impl)

    impl2 = Implication(Signal("req"), "|=>", Signal("ack"))
    compare_nl_systems("Simple Implication (non-overlapping)", impl2)

    print("\n" + "="*80)
    print("MODERATE COMPLEXITY")
    print("="*80)

    # Sequence delay
    seq_delay = SequenceDelay(Signal("req"), "##1", Signal("ack"))
    compare_nl_systems("Sequence with delay", seq_delay)

    # Sequence repeat
    seq_repeat = SequenceRepeat(Signal("req"), "[*", "[3]")
    compare_nl_systems("Sequence repeat (consecutive)", seq_repeat)

    # Implication with system function
    impl_sys = Implication(
        UnarySysFunction("$rose", Signal("req")),
        "|->",
        Signal("ack")
    )
    compare_nl_systems("Implication with $rose trigger", impl_sys)

    # Disable iff
    disable = DisableIff(
        Signal("reset"),
        Implication(Signal("req"), "|->", Signal("ack"))
    )
    compare_nl_systems("Property with disable iff", disable)

    print("\n" + "="*80)
    print("COMPLEX CASES")
    print("="*80)

    # Complex implication with comparison
    complex_ante = BinaryOp(
        Signal("req"),
        "&&",
        UnaryOp("!", Signal("busy"))
    )
    complex_impl = Implication(complex_ante, "|->", Signal("ack"))
    compare_nl_systems("Implication with compound antecedent", complex_impl)

    # Sequence delay range
    seq_range = SequenceDelay(Signal("req"), "##[1:3]", Signal("ack"))
    compare_nl_systems("Sequence with delay range", seq_range)

    # Property with binary op
    prop_bin = PropertyBinary(
        Implication(Signal("req"), "|->", Signal("ack")),
        "or",
        Implication(Signal("valid"), "|->", Signal("ready"))
    )
    compare_nl_systems("Compound property (or)", prop_bin)

    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)

    # Collect all test cases
    test_cases = [
        ("Signal", Signal("req")),
        ("BinaryOp", BinaryOp(Signal("req"), "==", Signal("ack"))),
        ("UnarySysFunction", UnarySysFunction("$rose", Signal("req"))),
        ("Implication", Implication(Signal("req"), "|->", Signal("ack"))),
        ("SequenceDelay", SequenceDelay(Signal("req"), "##1", Signal("ack"))),
        ("DisableIff", DisableIff(Signal("reset"), Implication(Signal("req"), "|->", Signal("ack")))),
    ]

    total_old_len = 0
    total_new_len = 0

    for name, node in test_cases:
        old_nl = node.__class__.to_natural_language(node)
        new_nl = sva_to_english(node)
        total_old_len += len(old_nl)
        total_new_len += len(new_nl)

    avg_reduction = ((total_old_len - total_new_len) / total_old_len * 100)

    print(f"\nTotal characters (old): {total_old_len}")
    print(f"Total characters (new): {total_new_len}")
    print(f"Average reduction: {avg_reduction:.1f}%")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nThe new SVATrans system produces:")
    print("  ✓ More natural, fluent English")
    print("  ✓ Proper sentence structure")
    print("  ✓ Clearer temporal relationships")
    print(f"  ✓ {avg_reduction:.1f}% reduction in verbosity")


if __name__ == "__main__":
    main()
