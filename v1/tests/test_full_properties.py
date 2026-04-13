#!/usr/bin/env python3
"""
Example usage of the improved SVA Implication Checker.

This demonstrates how the checker handles various SVA formats including:
- Simple inline expressions
- Full property declarations with property/endproperty
- Properties with $past, $rose, $fell, etc.
- Assert/assume/cover property wrappers
"""

import sys

from sva_toolkit.implication_checker.checker import SVAImplicationChecker, ImplicationResult


def example_simple():
    """Example with simple inline SVA expressions."""
    print("=" * 60)
    print("Example 1: Simple Inline SVA Expressions (with EBMC)")
    print("=" * 60)
    
    checker = SVAImplicationChecker(verbose=False)
    
    # A stronger consequent (more specific)
    sva_strong = "req |-> ##1 (gnt && !error)"
    # A weaker consequent (less specific)  
    sva_weak = "req |-> ##1 gnt"
    
    print(f"SVA Strong: {sva_strong}")
    print(f"SVA Weak:   {sva_weak}")
    print()
    
    # Strong should imply weak
    result = checker.check_implication(sva_strong, sva_weak)
    print(f"Strong -> Weak: {result.result.value}")
    print(f"Message: {result.message}")
    print()


def example_full_property():
    """Example with full property declarations."""
    print("=" * 60)
    print("Example 2: Full Property Declarations")
    print("=" * 60)
    
    checker = SVAImplicationChecker(verbose=False)
    
    # Full property with property/endproperty and $past
    sva1 = """property p_o_wb_dat_assignment;
    @(posedge i_clk)
    start_access == 1 |-> ##1 o_wb_dat == $past(i_write_data);
endproperty
assert_p_o_wb_dat_assignment: assert property (p_o_wb_dat_assignment) else $error("Assertion failed");"""

    # Simpler variant without the strict equality on start_access
    sva2 = """property p_o_wb_dat_simple;
    @(posedge i_clk)
    start_access |-> ##1 o_wb_dat == $past(i_write_data);
endproperty"""

    print("SVA1 (stronger - requires start_access == 1):")
    print(sva1[:100] + "...")
    print()
    print("SVA2 (weaker - only requires start_access to be true):")
    print(sva2[:100] + "...")
    print()
    
    # First, let's see what the checker extracts
    print("Extracted property body from SVA1:")
    print(f"  {checker.normalize_sva(sva1)}")
    print()
    print("Extracted property body from SVA2:")
    print(f"  {checker.normalize_sva(sva2)}")
    print()

    result = checker.check_implication(sva2, sva1, verbose=True)
    print(f"SVA2 -> SVA1: {result.result.value}")
    print(f"Message: {result.message}")
    print()


def example_with_builtin_functions():
    """Example with SVA built-in functions."""
    print("=" * 60)
    print("Example 3: Properties with Built-in Functions")
    print("=" * 60)
    
    checker = SVAImplicationChecker(verbose=False)
    
    # Property using $rose
    sva_rose = """property p_req_gnt;
    @(posedge clk)
    $rose(req) |-> ##[1:3] gnt;
endproperty"""

    # Property using explicit edge detection
    sva_explicit = """property p_req_gnt_explicit;
    @(posedge clk)
    (!$past(req) && req) |-> ##[1:3] gnt;
endproperty"""
    
    print("SVA with $rose:")
    print(f"  Normalized: {checker.normalize_sva(sva_rose)}")
    print()
    print("SVA with explicit edge detection:")
    print(f"  Normalized: {checker.normalize_sva(sva_explicit)}")
    print()
    result = checker.check_implication(sva_rose, sva_explicit)
    print(f"SVA with $rose -> SVA with explicit edge detection: {result.result.value}")
    print(f"Message: {result.message}")
    print()

def example_normalization():
    """Example showing normalization of different SVA formats."""
    print("=" * 60)
    print("Example 4: SVA Normalization")
    print("=" * 60)
    
    checker = SVAImplicationChecker(verbose=False)
    
    test_cases = [
        # Simple inline
        "req |-> ##1 gnt",
        
        # With clock
        "@(posedge clk) req |-> ##1 gnt",
        
        # Full property
        """property p_test;
            @(posedge clk)
            req |-> ##1 gnt;
        endproperty""",
        
        # Assert property wrapper
        "assert property (@(posedge clk) req |-> ##1 gnt);",
        
        # With disable iff
        """property p_test;
            @(posedge clk) disable iff (!rst_n)
            req |-> ##1 gnt;
        endproperty""",
        
        # Labeled assert with else
        """my_assert: assert property (p_test) else $error("Failed");""",
        
        # Complex with $past
        """property p_data;
            @(posedge clk) disable iff (!rst_n)
            valid |-> ##1 data_out == $past(data_in);
        endproperty
        assert property (p_data);""",
    ]
    
    for i, sva in enumerate(test_cases):
        print(f"Test Case {i+1}:")
        print(f"  Input:      {sva[:60]}{'...' if len(sva) > 60 else ''}")
        normalized = checker.normalize_sva(sva)
        print(f"  Normalized: {normalized}")
        print()


def example_signal_extraction():
    """Example showing signal extraction from SVA."""
    print("=" * 60)
    print("Example 5: Signal Extraction")
    print("=" * 60)
    
    checker = SVAImplicationChecker(verbose=False)
    
    sva = """property p_complex;
    @(posedge clk) disable iff (!rst_n)
    ($rose(req) && valid && mode == 2'b01) |-> 
        ##[1:5] (ack && data_out == $past(data_in, 2));
endproperty"""
    
    print("SVA:")
    print(sva)
    print()
    
    # Extract property body
    expr = checker.normalize_sva(sva)
    print(f"Property body: {expr}")
    print()
    
    # Extract signals
    signals = checker._collect_signals_from_expression(expr)
    print(f"Extracted signals: {signals}")
    print()
    
    # Analyze $past usage
    past_usage = checker._analyze_past_usage(expr)
    print(f"$past usage: {past_usage}")
    print()


def main():
    """Run all examples."""
    # These examples work without EBMC - they only use parsing/normalization
    example_normalization()
    example_signal_extraction()
    example_full_property()
    example_with_builtin_functions()
    
    # Check if EBMC is available for verification examples
    checker = SVAImplicationChecker(verbose=False)
    if checker.ebmc_available:
        example_simple()
    else:
        print("=" * 60)
        print("Note: EBMC is not installed.")
        print("The verification examples are skipped.")
        print("Install EBMC from: https://github.com/diffblue/hw-cbmc")
        print("=" * 60)
    
    print()
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()