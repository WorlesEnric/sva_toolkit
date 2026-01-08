#!/usr/bin/env python3
"""
SVAD Generation Demo

This example demonstrates how to generate SVA properties with
natural language descriptions (SVAD) using the sva_toolkit.
"""

from sva_toolkit.gen import SVASynthesizer, SVAProperty
from sva_toolkit.gen.types_sva import *
import random
import json

def demo_basic_generation():
    """Generate SVA-SVAD pairs programmatically."""
    print("=" * 80)
    print("Demo 1: Basic SVA-SVAD Generation")
    print("=" * 80)

    random.seed(42)
    synthesizer = SVASynthesizer(
        signals=['req', 'ack', 'gnt', 'valid'],
        max_depth=2,
        clock_signal='clk'
    )

    # Generate a single property
    prop = synthesizer.synthesize('p_handshake')

    print(f"\nProperty Name: {prop.name}")
    print(f"\nSVA Code:\n  {prop.sva_code}")
    print(f"\nSVAD (Natural Language):\n  {prop.svad}")
    print(f"\nFull Property Block:\n{prop.property_block}")


def demo_manual_construction():
    """Manually construct SVA nodes and get SVAD."""
    print("\n" + "=" * 80)
    print("Demo 2: Manual SVA Construction with SVAD")
    print("=" * 80)

    # Example 1: Simple implication
    req = Signal("req")
    ack = Signal("ack")
    rose_req = UnarySysFunction("$rose", req)
    impl = Implication(rose_req, "|=>", ack)

    print("\n--- Example 1: Request-Acknowledge ---")
    print(f"SVA:  {impl}")
    print(f"SVAD: {impl.to_natural_language()}")

    # Example 2: Sequence with delay
    valid = Signal("valid")
    ready = Signal("ready")
    seq = SequenceDelay(valid, "##[1:3]", ready)
    impl2 = Implication(req, "|->", seq)

    print("\n--- Example 2: Delayed Sequence ---")
    print(f"SVA:  {impl2}")
    print(f"SVAD: {impl2.to_natural_language()}")

    # Example 3: Complex property with past
    data = Signal("data")
    past_data = PastFunction(data, depth=2)
    comparison = BinaryOp(data, "==", past_data)
    stable_check = Implication(valid, "|->", comparison)

    print("\n--- Example 3: Data Stability Check ---")
    print(f"SVA:  {stable_check}")
    print(f"SVAD: {stable_check.to_natural_language()}")


def demo_json_export():
    """Generate and export SVA-SVAD pairs as JSON."""
    print("\n" + "=" * 80)
    print("Demo 3: JSON Export of SVA-SVAD Pairs")
    print("=" * 80)

    random.seed(123)
    synthesizer = SVASynthesizer(
        signals=['clk', 'rst_n', 'valid', 'ready', 'data'],
        max_depth=2
    )

    # Generate multiple properties
    properties = []
    for i in range(3):
        prop = synthesizer.synthesize(f'p_{i}')
        properties.append({
            'name': prop.name,
            'sva': prop.sva_code,
            'svad': prop.svad
        })

    # Export as JSON
    output = {
        'description': 'Auto-generated SVA properties with natural language descriptions',
        'properties': properties
    }

    print("\nGenerated JSON:")
    print(json.dumps(output, indent=2))


def demo_all_operators():
    """Demonstrate SVAD for various SVA operators."""
    print("\n" + "=" * 80)
    print("Demo 4: SVAD for Various SVA Constructs")
    print("=" * 80)

    req = Signal("req")
    ack = Signal("ack")
    valid = Signal("valid")

    examples = [
        ("Rose detection", UnarySysFunction("$rose", req)),
        ("Fell detection", UnarySysFunction("$fell", ack)),
        ("Stable check", UnarySysFunction("$stable", valid)),
        ("Changed detection", UnarySysFunction("$changed", req)),
        ("Past value", PastFunction(ack, 3)),
        ("Sequence delay", SequenceDelay(req, "##2", ack)),
        ("Repetition [*]", SequenceRepeat(valid, "[*", "3]")),
        ("Repetition [=]", SequenceRepeat(req, "[=", "2:4]")),
        ("Overlapping |->", Implication(req, "|->", ack)),
        ("Non-overlapping |=>", Implication(req, "|=>", ack)),
        ("Disable iff", DisableIff(Signal("rst_n"), Implication(req, "|->", ack))),
        ("Property NOT", NotProperty(valid)),
        ("Until", PropertyUntil(req, "until", ack)),
    ]

    for name, construct in examples:
        print(f"\n--- {name} ---")
        print(f"SVA:  {construct}")
        print(f"SVAD: {construct.to_natural_language()}")


if __name__ == "__main__":
    demo_basic_generation()
    demo_manual_construction()
    demo_json_export()
    demo_all_operators()

    print("\n" + "=" * 80)
    print("All demos complete!")
    print("=" * 80)
