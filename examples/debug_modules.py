#!/usr/bin/env python3
"""Debug script to output generated modules for SVA equivalence check."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sva_toolkit.implication_checker.checker import SVAImplicationChecker

def main():
    """Check the specific SVAs and output generated modules."""
    sva1 = "@(posedge clk_primary)    $rose(clk_switch_req) |-> ##1 $fell(clk_primary) ##0 $rose(clk_secondary)"
    sva2 = "@(posedge clk_primary)     $rose(clk_switch_req) |-> ##1 ($fell(clk_primary) && $rose(clk_secondary))"
    
    print("=" * 80)
    print("SVA 1:", sva1)
    print("SVA 2:", sva2)
    print("=" * 80)
    print()
    
    checker = SVAImplicationChecker(verbose=False)
    
    print("Checking equivalence...")
    result = checker.check_equivalence(sva1, sva2)
    
    print()
    print("=" * 80)
    print("RESULT:", result.result.value)
    print("MESSAGE:", result.message)
    print("=" * 80)
    print()
    
    if result.module:
        print("=" * 80)
        print("GENERATED MODULES:")
        print("=" * 80)
        print(result.module)
        print("=" * 80)
    else:
        print("No modules generated (error occurred before module generation)")

if __name__ == "__main__":
    main()

